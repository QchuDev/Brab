---
inclusion: manual
---

# Auditoría semanal de la librería Rust

Objetivo: detectar problemas de calidad, compatibilidad y seguridad, y reportarlos como issues en GitHub sin duplicar los existentes.

## Chequeos (en orden)

1. `cargo fmt --check`
2. `cargo clippy --all-targets --all-features -- -D warnings` (toolchain stable)
3. `cargo test --all-features` (toolchain stable)
4. `cargo +MSRV check --all-targets`, donde MSRV es el `rust-version` de `Cargo.toml` (actualmente 1.89)
5. `cargo audit` (vulnerabilidades en dependencias)
6. `cargo outdated --root-deps-only` (dependencias desactualizadas)

## Reglas

- Ejecuta cada comando solo, sin pipes (`|`), `;`, `&&`, redirecciones ni `echo`. Si la salida es larga, léela completa y resúmela tú. El shell solo auto-aprueba comandos simples.
- No modifiques código ni hagas commits. Solo lees, ejecutas chequeos y creas issues.
- Antes de crear un issue, ejecuta `gh issue list --state open --limit 100` y no abras uno si ya existe uno equivalente.
- Un issue por problema (agrupa los warnings de un mismo tipo). Título con prefijo `[audit]`.
- Cada issue incluye: comando ejecutado, salida relevante, archivo y línea si aplica, y una sugerencia de arreglo.
- Usa `gh issue create --title ... --body ...` con la etiqueta `kiro-audit` si existe.
- Si todo pasa, no crees nada y termina con un resumen de una línea por chequeo.
- Termina siempre con un resumen: chequeos ejecutados, resultado y URLs de issues creados.
