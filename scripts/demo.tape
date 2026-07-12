# Proctor — Agent trajectory evaluation CLI (real terminal recording)
# Generated with VHS (charmbracelet/vhs)

Output docs/demo.gif
Set Width 1280
Set Height 720
Set FontFamily "DejaVu Sans Mono"
Set FontSize 16
Set Padding 0
Set Margin 0
Set LineHeight 1.4
Set Theme "Chalk"
Set Framerate 15

# Scene 1: eval with markdown report
Type "proctor eval --dataset examples/golden.json"
Sleep 400ms
Enter
Sleep 1.5s

Ctrl+L
Sleep 300ms

# Scene 2: eval to JSON report (CI artifact)
Type "proctor eval --dataset examples/golden.json --format json --out report.json"
Sleep 400ms
Enter
Sleep 1.5s

Type "ls -la report.json"
Sleep 300ms
Enter
Sleep 1s

Ctrl+L
Sleep 300ms

# Scene 3: CI gate
Type "proctor gate --current report.json"
Sleep 400ms
Enter
Sleep 1.5s

Ctrl+L
Sleep 300ms

# Scene 4: diff v1 vs v2
Type "proctor eval --dataset examples/golden.json --format json --out report_v1.json"
Sleep 300ms
Enter
Sleep 1.5s

Type "proctor diff --base report_v1.json --current report_v1.json"
Sleep 300ms
Enter
Sleep 1.5s

Ctrl+L
Sleep 300ms

# Scene 5: attribution in action — show a failing sample
Type "proctor eval --dataset examples/golden.json --format markdown | head -25"
Sleep 400ms
Enter
Sleep 2s

Ctrl+L
Sleep 300ms

# Footer
Type "proctor -- eval, gate, diff, align, compare, serve"
Sleep 1.2s
