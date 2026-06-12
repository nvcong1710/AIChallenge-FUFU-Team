# sync_docs.ps1 — copy các file md tài liệu dự án ở repo root vào docs/du-an/
# để website docs (docs/index.html) đọc được khi deploy GitHub Pages từ /docs.
# Chạy lại script này MỖI KHI sửa các file md gốc:  .\scripts\sync_docs.ps1

$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "docs\du-an"
New-Item -ItemType Directory -Force $dest | Out-Null

$files = @(
    "PROJECT-CONTEXT.md",
    "RESEARCH-PLAN.md",
    "README-V2.md",
    "HARDWARE.md",
    "KIEN-TRUC-MERMAID.md",
    "BAO-CAO-TONG-HOP.md",
    "CHECKLIST-GIAI-PHAP.md",
    "VAST_AI_GUIDE.md"
)

foreach ($f in $files) {
    $src = Join-Path $root $f
    if (Test-Path $src) {
        Copy-Item $src $dest -Force
        Write-Host "synced  $f"
    } else {
        Write-Host "MISSING $f" -ForegroundColor Yellow
    }
}
Write-Host "`nDone -> $dest"
