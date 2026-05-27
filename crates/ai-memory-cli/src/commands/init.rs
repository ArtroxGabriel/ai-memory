//! `ai-memory init` — create the data directory layout.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli::InitArgs;
use crate::config::Config;

const DEFAULT_CONFIG_TOML: &str = include_str!("../../templates/config.default.toml");

const SUBDIRS: &[&str] = &["wiki", "raw", "db", "models"];

/// Run the `init` subcommand.
///
/// Creates `<data_dir>/{wiki,raw,db,models}` (idempotent) and writes a default
/// config file unless one already exists (use `--force` to overwrite). With no
/// explicit `--config`, the config remains at `<data_dir>/config.toml`; packaged
/// systemd units can pass `--config /etc/ai-memory/config.toml` or
/// `--config ~/.config/ai-memory/config.toml` without changing the data root.
///
/// # Errors
/// Returns an error if directories cannot be created or the config file
/// cannot be written.
pub fn run(config: &Config, args: InitArgs, config_path: Option<&Path>) -> Result<()> {
    let root = &config.data_dir;
    fs::create_dir_all(root).with_context(|| format!("creating data root {}", root.display()))?;

    for sub in SUBDIRS {
        let path = root.join(sub);
        fs::create_dir_all(&path).with_context(|| format!("creating {}", path.display()))?;
        tracing::info!(path = %path.display(), "ensured directory");
    }

    let cfg_path = init_config_path(root, config_path);
    if cfg_path.exists() && !args.force {
        tracing::info!(
            path = %cfg_path.display(),
            "config already exists; leaving untouched (pass --force to overwrite)",
        );
    } else {
        if let Some(parent) = cfg_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating config dir {}", parent.display()))?;
        }
        let mut f = fs::File::create(&cfg_path)
            .with_context(|| format!("creating {}", cfg_path.display()))?;
        f.write_all(DEFAULT_CONFIG_TOML.as_bytes())
            .with_context(|| format!("writing {}", cfg_path.display()))?;
        tracing::info!(path = %cfg_path.display(), "wrote default config");
    }

    tracing::info!("init complete");
    Ok(())
}

fn init_config_path(data_dir: &Path, explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn cfg_in(dir: &std::path::Path) -> Config {
        Config {
            data_dir: dir.to_path_buf(),
            ..Config::default()
        }
    }

    #[test]
    fn init_creates_subdirs_and_config() {
        let tmp = TempDir::new().unwrap();
        let config = cfg_in(tmp.path());
        run(&config, InitArgs { force: false }, None).unwrap();
        for sub in SUBDIRS {
            assert!(tmp.path().join(sub).is_dir(), "missing {sub}");
        }
        assert!(tmp.path().join("config.toml").exists());
    }

    #[test]
    fn init_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let config = cfg_in(tmp.path());
        run(&config, InitArgs { force: false }, None).unwrap();
        // Touch the config to detect a clobber.
        let stamp = std::fs::metadata(tmp.path().join("config.toml"))
            .unwrap()
            .modified()
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        run(&config, InitArgs { force: false }, None).unwrap();
        let stamp2 = std::fs::metadata(tmp.path().join("config.toml"))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(stamp, stamp2, "second init clobbered the config");
    }

    #[test]
    fn init_honors_explicit_config_path_outside_data_dir() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let config_dir = tmp.path().join("config").join("ai-memory");
        let config_path = config_dir.join("config.toml");
        let config = cfg_in(&data_dir);

        run(
            &config,
            InitArgs { force: false },
            Some(config_path.as_path()),
        )
        .unwrap();

        assert!(data_dir.join("wiki").is_dir());
        assert!(data_dir.join("raw").is_dir());
        assert!(config_path.is_file());
        assert!(
            !data_dir.join("config.toml").exists(),
            "explicit --config must not also write data-dir config"
        );
    }
}
