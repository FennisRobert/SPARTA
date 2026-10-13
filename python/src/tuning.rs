//! The optimizer and the saved profiles, for Python.
//!
//! Everything here is a thin shell over `sparta::optimize` and
//! `sparta::profiles`: settings cross as dicts keyed by the shared setting
//! names, and the search, its progress text and its summary are the Rust
//! ones, so Python and Rust report the same run in the same words.
use crate::error;
use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyKeyError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyString},
};
use sparta::optimize::{self, Backend, Options, Outcome, Problem};
use sparta::profiles;
use sparta::settings::Value;
use sparta::Config;
use std::time::Duration;

/// Every setting of `config` as a dict keyed by setting name.
pub(crate) fn settings(py: Python<'_>, config: &Config) -> PyResult<Py<PyDict>> {
    let out = PyDict::new_bound(py);
    for (name, value) in config.settings() {
        out.set_item(name, value_to_py(py, &value))?;
    }
    Ok(out.unbind())
}

/// The default Config with every entry of `settings` applied, checked.
pub(crate) fn config_from(settings: &Bound<'_, PyDict>) -> PyResult<Config> {
    let mut config = Config::default();
    for (key, value) in settings.iter() {
        let name: String = key.extract()?;
        config.set(&name, value_from_py(&value)?).map_err(PyValueError::new_err)?;
    }
    config.check().map_err(PyValueError::new_err)?;
    Ok(config)
}

fn value_to_py(py: Python<'_>, value: &Value) -> PyObject {
    match value {
        Value::Bool(v) => v.to_object(py),
        Value::Int(v) => v.to_object(py),
        Value::Float(v) => v.to_object(py),
        Value::Text(v) => v.to_object(py),
        Value::None => py.None(),
    }
}

fn value_from_py(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    // bool first: in Python it is a kind of int.
    if value.is_none() {
        Ok(Value::None)
    } else if value.is_instance_of::<PyBool>() {
        Ok(Value::Bool(value.extract()?))
    } else if value.is_instance_of::<PyInt>() {
        Ok(Value::Int(value.extract().map_err(|_| PyValueError::new_err("settings cannot be negative"))?))
    } else if value.is_instance_of::<PyFloat>() {
        Ok(Value::Float(value.extract()?))
    } else if value.is_instance_of::<PyString>() {
        Ok(Value::Text(value.extract()?))
    } else {
        Err(PyTypeError::new_err(format!("a setting cannot be {}", value.get_type().name()?)))
    }
}

/// A finished optimization: the tuned settings and what they gained.
#[pyclass(name = "_Outcome", frozen)]
pub(crate) struct PyOutcome {
    outcome: Outcome,
}

#[pymethods]
impl PyOutcome {
    /// The tuned settings, keyed by setting name.
    fn config(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        settings(py, &self.outcome.config)
    }
    /// `"gpu"` or `"cpu"`: the backend the tuned settings select.
    #[getter]
    fn backend(&self) -> &'static str {
        self.outcome.backend.name()
    }
    /// How many times faster than the starting settings.
    #[getter]
    fn speedup(&self) -> f64 {
        self.outcome.speedup()
    }
    /// Per backend: (name, start seconds, tuned seconds).
    #[getter]
    fn seconds(&self) -> Vec<(&'static str, f64, f64)> {
        self.outcome.seconds.iter().map(|&(b, s, t)| (b.name(), s, t)).collect()
    }
    /// Every setting that moved: (name, start value, tuned value).
    fn changes(&self, py: Python<'_>) -> Vec<(&'static str, PyObject, PyObject)> {
        self.outcome
            .changes()
            .into_iter()
            .map(|(n, a, b)| (n, value_to_py(py, &a), value_to_py(py, &b)))
            .collect()
    }
    /// Every candidate: (backend, setting, value, seconds or None, note).
    fn trials(&self, py: Python<'_>) -> Vec<(&'static str, &'static str, PyObject, Option<f64>, String)> {
        self.outcome
            .trials
            .iter()
            .map(|t| (t.backend.name(), t.setting, value_to_py(py, &t.value), t.seconds, t.note.clone()))
            .collect()
    }
    /// Iterations run, and asked for.
    #[getter]
    fn iterations(&self) -> (usize, usize) {
        self.outcome.iterations
    }
    #[getter]
    fn elapsed(&self) -> f64 {
        self.outcome.elapsed.as_secs_f64()
    }
    /// Saves the result as profile `name`, with its measurements.
    #[pyo3(signature = (name, note = ""))]
    fn save(&self, name: &str, note: &str) -> PyResult<()> {
        let mut profile = self.outcome.profile();
        profile.note = note.to_string();
        profiles::save(name, &profile).map_err(error)
    }
    fn __str__(&self) -> String {
        self.outcome.to_string()
    }
}

/// Runs `sparta::optimize::optimize` on prepared problems.
///
/// Each problem is `(n, ptr, rows, values, rhs, lu)`: CSC exactly as the
/// analysis takes it, and `rhs` an `(n, k)` array or None. `progress` is
/// called with `(line, done, total, elapsed, remaining)` after every step, or
/// is None. Ctrl-C, or an exception raised by `progress`, stops the search
/// and is raised here.
#[pyfunction]
#[pyo3(signature = (problems, start, gpu, cpu, analysis, time_limit, iterations, min_gain, progress))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn _optimize(
    py: Python<'_>,
    problems: Vec<(
        usize,
        PyReadonlyArray1<'_, usize>,
        PyReadonlyArray1<'_, usize>,
        PyReadonlyArray1<'_, Complex64>,
        Option<PyReadonlyArray2<'_, Complex64>>,
        bool,
    )>,
    start: &Bound<'_, PyDict>,
    gpu: bool,
    cpu: bool,
    analysis: bool,
    time_limit: Option<f64>,
    iterations: usize,
    min_gain: f64,
    progress: Option<PyObject>,
) -> PyResult<PyOutcome> {
    let mut prepared = Vec::with_capacity(problems.len());
    for (n, ptr, rows, values, rhs, lu) in problems {
        let (ptr, rows, values) = (ptr.as_slice()?.to_vec(), rows.as_slice()?.to_vec(), values.as_slice()?.to_vec());
        let mut problem = if lu {
            Problem::lu(n, ptr, rows, values)
        } else {
            Problem::ldlt(n, ptr, rows, values)
        }
        .map_err(error)?;
        if let Some(rhs) = rhs {
            let block = rhs.as_array();
            let nrhs = block.ncols();
            // Columns end to end, as the solver packs right-hand sides.
            let packed: Vec<Complex64> = block.t().iter().copied().collect();
            problem = problem.with_rhs(packed, nrhs).map_err(error)?;
        }
        prepared.push(problem);
    }
    if prepared.is_empty() {
        prepared = Problem::standard();
    }
    let options = Options {
        start: config_from(start)?,
        gpu,
        cpu,
        analysis,
        time_limit: time_limit.map(Duration::from_secs_f64),
        iterations,
        min_gain,
    };
    let mut raised: Option<PyErr> = None;
    let result = optimize::optimize(&prepared, &options, &mut |p| {
        let step = py.check_signals().and_then(|()| match &progress {
            Some(callback) => callback
                .call1(
                    py,
                    (
                        p.to_string(),
                        p.done,
                        p.total,
                        p.elapsed.as_secs_f64(),
                        p.remaining.map(|r| r.as_secs_f64()),
                    ),
                )
                .map(|_| ()),
            None => Ok(()),
        });
        match step {
            Ok(()) => true,
            Err(e) => {
                raised = Some(e);
                false
            }
        }
    });
    if let Some(e) = raised {
        return Err(e);
    }
    Ok(PyOutcome { outcome: result.map_err(error)? })
}

/// Saves `settings` as profile `name`, without measurements.
#[pyfunction]
#[pyo3(signature = (name, settings, note = ""))]
pub(crate) fn _profile_save(name: &str, settings: &Bound<'_, PyDict>, note: &str) -> PyResult<()> {
    let mut profile = profiles::Profile::new(config_from(settings)?);
    profile.note = note.to_string();
    profiles::save(name, &profile).map_err(error)
}

/// Returns profile `name` as (settings, chip it was made on, whether that is
/// another chip). KeyError when there is none.
#[pyfunction]
pub(crate) fn _profile_load(py: Python<'_>, name: &str) -> PyResult<(Py<PyDict>, String, bool)> {
    if !profiles::list().map_err(error)?.contains_key(name) {
        return Err(PyKeyError::new_err(format!(
            "no saved profile '{name}' in {}",
            profiles::path().display()
        )));
    }
    let profile = profiles::load(name).map_err(error)?;
    Ok((settings(py, &profile.config)?, profile.machine.chip.clone(), profile.foreign()))
}

/// Removes profile `name`; KeyError when there is none.
#[pyfunction]
pub(crate) fn _profile_delete(name: &str) -> PyResult<()> {
    if !profiles::list().map_err(error)?.contains_key(name) {
        return Err(PyKeyError::new_err(format!(
            "no saved profile '{name}' in {}",
            profiles::path().display()
        )));
    }
    profiles::delete(name).map_err(error)
}

/// Every saved profile as (name, one-line summary).
#[pyfunction]
pub(crate) fn _profile_list() -> PyResult<Vec<(String, String)>> {
    Ok(profiles::list()
        .map_err(error)?
        .into_iter()
        .map(|(name, p)| {
            let backend = if p.config.force_cpu { Backend::Cpu } else { Backend::Gpu };
            let mut parts = vec![p.machine.chip.clone(), backend.to_string()];
            if let Some(t) = &p.tuning {
                parts.push(format!("{:.2}x over defaults", t.speedup));
            }
            parts.push(p.created.clone());
            if !p.note.is_empty() {
                parts.push(p.note.clone());
            }
            (name, parts.join(", "))
        })
        .collect())
}

/// Raises ValueError unless `name` can name a profile.
#[pyfunction]
pub(crate) fn _profile_check(name: &str) -> PyResult<()> {
    profiles::check_name(name).map(|_| ()).map_err(error)
}

/// The profile file's path.
#[pyfunction]
pub(crate) fn _profile_path() -> String {
    profiles::path().display().to_string()
}
