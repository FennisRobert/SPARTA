//! Named configurations saved for later: `configs.json` in the SPARTA directory.
//!
//! A profile is a [`Config`] stored under a name, usually by
//! [`optimize`](crate::optimize::optimize), with a record of where and how it
//! was made: the machine, the SPARTA version and, for a tuned profile, what it
//! gained. Only the settings are needed to use it; the rest lets a profile
//! carried to another computer, or kept across an upgrade, say so.
//!
//! The file lives in the SPARTA directory, `~/Library/SPARTA` on macOS and
//! `~/.sparta` elsewhere.
//! `SPARTA_CONFIG` names another file instead. A save replaces the file
//! atomically, so an interrupted one never leaves it half written.
//!
//! Settings are stored by their [`settings`](crate::settings) names. On load,
//! a setting missing from an older file takes today's default and one this
//! version no longer has is ignored, so profiles survive upgrades.

use crate::config::Config;
use crate::settings::{Value, NAMES};
use crate::{Error, Preset, Result};
use serde_json::{json, Map, Value as Json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

/// The layout version written into the file.
const FORMAT: u64 = 1;

/// A saved configuration and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub config: Config,
    /// Free text shown when profiles are listed.
    pub note: String,
    /// The day it was saved, as `YYYY-MM-DD`.
    pub created: String,
    /// The SPARTA version that saved it.
    pub version: String,
    /// The machine it was saved on.
    pub machine: Machine,
    /// What the optimizer measured, for a tuned profile.
    pub tuning: Option<Tuning>,
}

/// Enough about a computer to notice a profile made on a different one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    /// The processor, such as `Apple M4 Pro`; on Apple silicon also the GPU.
    pub chip: String,
    pub os: String,
}

/// What [`optimize`](crate::optimize::optimize) measured for a profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    /// `"gpu"` or `"cpu"`: the backend the profile selects.
    pub backend: String,
    /// Default seconds over tuned seconds.
    pub speedup: f64,
    pub default_seconds: f64,
    pub tuned_seconds: f64,
    /// Whether analysis time was counted.
    pub analysis: bool,
    /// Order and stored entries of each problem it was tuned on.
    pub problems: Vec<(usize, usize)>,
}

impl Profile {
    /// A profile of `config`, dated today on this machine.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            note: String::new(),
            created: today(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            machine: machine(),
            tuning: None,
        }
    }

    /// True when this profile was saved on a different chip.
    pub fn foreign(&self) -> bool {
        self.machine.chip != machine().chip
    }

    fn to_json(&self) -> Json {
        let settings: Map<String, Json> = self
            .config
            .settings()
            .into_iter()
            .map(|(name, value)| (name.to_string(), to_json(&value)))
            .collect();
        let mut out = json!({
            "config": settings,
            "note": self.note,
            "created": self.created,
            "sparta": self.version,
            "machine": {"chip": self.machine.chip, "os": self.machine.os},
        });
        if let Some(t) = &self.tuning {
            out["tuning"] = json!({
                "backend": t.backend,
                "speedup": t.speedup,
                "default_seconds": t.default_seconds,
                "tuned_seconds": t.tuned_seconds,
                "analysis": t.analysis,
                "problems": t.problems.iter().map(|&(n, nnz)| json!({"n": n, "nnz": nnz})).collect::<Vec<_>>(),
            });
        }
        out
    }

    fn from_json(name: &str, value: &Json) -> Result<Self> {
        let text = |v: &Json, key: &str| v.get(key).and_then(Json::as_str).unwrap_or("").to_string();
        let mut config = Config::default();
        let stored = value
            .get("config")
            .and_then(Json::as_object)
            .ok_or_else(|| Error::Invalid(format!("profile '{name}' has no settings")))?;
        for name in NAMES {
            if let Some(v) = stored.get(name) {
                config
                    .set(name, from_json(v))
                    .map_err(|e| Error::Invalid(format!("profile '{name}': {e}")))?;
            }
        }
        let machine = value.get("machine").unwrap_or(&Json::Null);
        let tuning = value.get("tuning").filter(|t| t.is_object()).map(|t| {
            let number = |key: &str| t.get(key).and_then(Json::as_f64).unwrap_or(f64::NAN);
            Tuning {
                backend: text(t, "backend"),
                speedup: number("speedup"),
                default_seconds: number("default_seconds"),
                tuned_seconds: number("tuned_seconds"),
                analysis: t.get("analysis").and_then(Json::as_bool).unwrap_or(false),
                problems: t
                    .get("problems")
                    .and_then(Json::as_array)
                    .map(|ps| {
                        ps.iter()
                            .map(|p| {
                                let get = |k| p.get(k).and_then(Json::as_u64).unwrap_or(0) as usize;
                                (get("n"), get("nnz"))
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            }
        });
        Ok(Self {
            config,
            note: text(value, "note"),
            created: text(value, "created"),
            version: text(value, "sparta"),
            machine: Machine {
                chip: text(machine, "chip"),
                os: text(machine, "os"),
            },
            tuning,
        })
    }
}

/// The profile file, whether or not it exists yet.
pub fn path() -> PathBuf {
    if let Some(file) = std::env::var_os("SPARTA_CONFIG").filter(|f| !f.is_empty()) {
        return PathBuf::from(file);
    }
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let dir = if cfg!(target_os = "macos") { "Library/SPARTA" } else { ".sparta" };
    home.join(dir).join("configs.json")
}

/// Every saved profile by name. A missing file holds none.
pub fn list() -> Result<BTreeMap<String, Profile>> {
    read()?
        .iter()
        .map(|(name, value)| Ok((name.clone(), Profile::from_json(name, value)?)))
        .collect()
}

/// The profile saved as `name`.
///
/// Its settings are checked against this build's device, so a profile whose
/// geometry this GPU cannot run is refused here rather than at analysis.
pub fn load(name: &str) -> Result<Profile> {
    let all = read()?;
    let value = all.get(name).ok_or_else(|| missing(name, &all))?;
    let profile = Profile::from_json(name, value)?;
    profile
        .config
        .check()
        .map_err(|e| Error::Invalid(format!("profile '{name}' does not suit this build: {e}")))?;
    Ok(profile)
}

/// Saves `profile` as `name`, replacing any profile of that name.
///
/// Preset names are refused, so that a name given where either is accepted
/// can never mean two things.
pub fn save(name: &str, profile: &Profile) -> Result<()> {
    let name = check_name(name)?;
    profile.config.check().map_err(Error::Invalid)?;
    let mut all = read()?;
    all.insert(name.to_string(), profile.to_json());
    write(all)
}

/// Removes the profile saved as `name`.
pub fn delete(name: &str) -> Result<()> {
    let mut all = read()?;
    if all.remove(name).is_none() {
        return Err(missing(name, &all));
    }
    write(all)
}

/// Returns `name` trimmed, or why it cannot name a profile.
pub fn check_name(name: &str) -> Result<&str> {
    let name = name.trim();
    if name.is_empty() {
        return Err(Error::Invalid("a profile name cannot be empty".into()));
    }
    if Preset::parse(name).is_some() {
        return Err(Error::Invalid(format!(
            "'{name}' is a built-in preset name; choose another profile name"
        )));
    }
    Ok(name)
}

/// Describes this computer.
pub fn machine() -> Machine {
    static MACHINE: std::sync::OnceLock<Machine> = std::sync::OnceLock::new();
    MACHINE
        .get_or_init(|| {
            let run = |cmd: &str, args: &[&str]| {
                std::process::Command::new(cmd)
                    .args(args)
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            };
            let chip = if cfg!(target_os = "macos") {
                run("sysctl", &["-n", "machdep.cpu.brand_string"])
            } else {
                fs::read_to_string("/proc/cpuinfo").ok().and_then(|info| {
                    info.lines()
                        .find(|l| l.starts_with("model name"))
                        .and_then(|l| l.split_once(':'))
                        .map(|(_, v)| v.trim().to_string())
                })
            };
            let os = if cfg!(target_os = "macos") {
                run("sw_vers", &["-productVersion"]).map(|v| format!("macOS {v}"))
            } else {
                None
            };
            Machine {
                chip: chip.unwrap_or_else(|| std::env::consts::ARCH.to_string()),
                os: os.unwrap_or_else(|| std::env::consts::OS.to_string()),
            }
        })
        .clone()
}

fn missing(name: &str, all: &Map<String, Json>) -> Error {
    let known: Vec<&str> = all.keys().map(String::as_str).collect();
    Error::Invalid(format!(
        "no saved profile '{name}' in {} (saved: {})",
        path().display(),
        if known.is_empty() { "none".to_string() } else { known.join(", ") }
    ))
}

fn read() -> Result<Map<String, Json>> {
    let path = path();
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Map::new()),
        Err(e) => return Err(Error::Invalid(format!("reading {}: {e}", path.display()))),
    };
    let data: Json = serde_json::from_str(&text)
        .map_err(|e| Error::Invalid(format!("{} is not valid JSON: {e}", path.display())))?;
    data.get("profiles")
        .and_then(Json::as_object)
        .cloned()
        .ok_or_else(|| Error::Invalid(format!("{} is not a SPARTA profile file", path.display())))
}

fn write(profiles: Map<String, Json>) -> Result<()> {
    let path = path();
    let fail = |e: std::io::Error| Error::Invalid(format!("writing {}: {e}", path.display()));
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(fail)?;
    }
    let text = serde_json::to_string_pretty(&json!({"format": FORMAT, "profiles": profiles}))
        .expect("a JSON value always serializes");
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::write(&temporary, text + "\n").map_err(fail)?;
    fs::rename(&temporary, &path).map_err(|e| {
        let _ = fs::remove_file(&temporary);
        fail(e)
    })
}

fn to_json(value: &Value) -> Json {
    match value {
        Value::Bool(v) => json!(v),
        Value::Int(v) => json!(v),
        Value::Float(v) => json!(v),
        Value::Text(v) => json!(v),
        Value::None => Json::Null,
    }
}

fn from_json(value: &Json) -> Value {
    match value {
        Json::Bool(v) => Value::Bool(*v),
        Json::Number(n) => n.as_u64().map_or_else(|| Value::Float(n.as_f64().unwrap_or(f64::NAN)), Value::Int),
        Json::String(s) => Value::Text(s.clone()),
        _ => Value::None,
    }
}

/// Today as `YYYY-MM-DD`, in UTC.
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    // Howard Hinnant's days-to-civil conversion.
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Save, list, load and delete round-trip through one file, and a preset
    /// name is refused.
    fn profiles_round_trip() {
        let file = std::env::temp_dir().join(format!("sparta-profiles-{}.json", std::process::id()));
        std::env::set_var("SPARTA_CONFIG", &file);
        let mut profile = Profile::new(Config {
            nemin: 32,
            ..Config::default()
        });
        profile.tuning = Some(Tuning {
            backend: "gpu".into(),
            speedup: 1.25,
            default_seconds: 1.0,
            tuned_seconds: 0.8,
            analysis: false,
            problems: vec![(1000, 7000)],
        });
        save("mine", &profile).unwrap();
        assert_eq!(load("mine").unwrap(), profile);
        assert_eq!(list().unwrap().len(), 1);
        assert!(save("sweep", &profile).is_err());
        delete("mine").unwrap();
        assert!(load("mine").is_err());
        assert!(delete("mine").is_err());
        std::env::remove_var("SPARTA_CONFIG");
        let _ = fs::remove_file(file);
        assert_eq!(today().len(), 10);
    }
}
