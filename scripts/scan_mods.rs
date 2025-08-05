use std::fs;
use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};

fn find_rs_files(base: &Path, files: &mut Vec<PathBuf>) {
    if base.is_dir() {
        for entry in fs::read_dir(base).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                find_rs_files(&path, files);
            } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
                files.push(path);
            }
        }
    }
}

fn main() {
    let base = Path::new("src");
    let mut files = Vec::new();
    find_rs_files(base, &mut files);

    // Map file names to paths
    let mut name_to_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();
    // Map file path to referenced mods and uses
    let mut mod_refs: HashMap<String, HashSet<String>> = HashMap::new();

    for file in &files {
        let content = fs::read_to_string(file).unwrap_or_default();
        let fname = file.file_name().unwrap().to_string_lossy().into_owned();
        name_to_paths.entry(fname.clone()).or_default().push(file.clone());

        let mut refs = HashSet::new();
        for line in content.lines() {
            if line.trim().starts_with("mod ") || line.trim().starts_with("use ") {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() > 1 {
                    refs.insert(tokens[1].replace(';', "").replace("crate::", ""));
                }
            }
        }
        mod_refs.insert(file.display().to_string(), refs);
    }

    println!("\n=== 📁 Duplicate File Names ===");
    for (name, paths) in &name_to_paths {
        if paths.len() > 1 {
            println!("\n{name}:");
            for p in paths {
                println!("  - {}", p.display());
            }
        }
    }

    println!("\n=== 🔗 Module References ===");
    for (file, refs) in &mod_refs {
        if !refs.is_empty() {
            println!("\n{} references:", file);
            for r in refs {
                println!("  - {r}");
            }
        }
    }

    println!("\n=== 🧾 Summary Table ===");
    println!("{:30} | {}", "File", "Locations");
    println!("{}", "-".repeat(60));
    for (name, paths) in &name_to_paths {
        println!("{:30} | {}", name, paths.len());
    }
}
