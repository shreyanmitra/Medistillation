# cleanup_old_files.ps1
# Script to delete old/duplicate files after reorganization

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Med-Distillation Cleanup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$baseDir = "c:\Users\mahasweta\Documents\GitHub\Medistillation"

# Ask for confirmation
Write-Host "This script will delete the following:" -ForegroundColor Yellow
Write-Host "  - Old info/ directory files" -ForegroundColor Yellow
Write-Host "  - Empty/duplicate files (main.py, DatasetBuilder.py, etc.)" -ForegroundColor Yellow
Write-Host "  - Root-level markdown files (moved to docs/)" -ForegroundColor Yellow
Write-Host "  - Old docs location files (moved to subdirectories)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Files have been COPIED to new locations first." -ForegroundColor Green
Write-Host ""

$confirm = Read-Host "Continue with cleanup? (yes/no)"

if ($confirm -ne "yes") {
    Write-Host "Cleanup cancelled." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "Starting cleanup..." -ForegroundColor Green
Write-Host ""

# Track deletions
$deleted = @()
$failed = @()

# Function to safely delete file
function Remove-FileSafely {
    param($path)
    try {
        if (Test-Path $path) {
            Remove-Item $path -Force
            Write-Host "  ✓ Deleted: $path" -ForegroundColor Green
            $script:deleted += $path
        } else {
            Write-Host "  ℹ Not found: $path" -ForegroundColor Gray
        }
    } catch {
        Write-Host "  ✗ Failed: $path - $_" -ForegroundColor Red
        $script:failed += $path
    }
}

# 1. Delete info/ directory files
Write-Host "[1/6] Cleaning up info/ directory..." -ForegroundColor Cyan
Remove-FileSafely "$baseDir\info\Code_Organization_Architecture.md"
Remove-FileSafely "$baseDir\info\EXPERIMENT_PROCEDURE.md"
Remove-FileSafely "$baseDir\info\LogitKD_Changes_Analysis.md"
Remove-FileSafely "$baseDir\info\LogitKD_Implementation_Guide.md"
Remove-FileSafely "$baseDir\info\PlanOfAction102625"
Remove-FileSafely "$baseDir\info\project_details_latex.tex"
Remove-FileSafely "$baseDir\info\references.bib"

# Remove empty info directory
if (Test-Path "$baseDir\info") {
    $infoFiles = Get-ChildItem "$baseDir\info" -Force
    if ($infoFiles.Count -eq 0) {
        Remove-Item "$baseDir\info" -Force
        Write-Host "  ✓ Removed empty directory: info/" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ info/ not empty, skipping removal" -ForegroundColor Yellow
    }
}

# 2. Delete empty/placeholder files
Write-Host ""
Write-Host "[2/6] Deleting empty/placeholder files..." -ForegroundColor Cyan
Remove-FileSafely "$baseDir\src\DatasetBuilder.py"
Remove-FileSafely "$baseDir\main.py"
Remove-FileSafely "$baseDir\sample_data.py"

# 3. Delete root-level markdown files (moved to docs/)
Write-Host ""
Write-Host "[3/6] Cleaning up root-level markdown files..." -ForegroundColor Cyan
Remove-FileSafely "$baseDir\DATASET_STRATEGY.md"
Remove-FileSafely "$baseDir\IMPLEMENTATION_COMPLETE.md"

# 4. Delete old docs/ location files (moved to subdirectories)
Write-Host ""
Write-Host "[4/6] Cleaning up old docs/ files..." -ForegroundColor Cyan
Remove-FileSafely "$baseDir\docs\EXPERIMENT_PROCEDURE.md"
Remove-FileSafely "$baseDir\docs\FIDELITYBENCH_GUIDE.md"
Remove-FileSafely "$baseDir\docs\VISUALIZATION_GUIDE.md"
Remove-FileSafely "$baseDir\docs\VISUALIZATION_FEATURES.md"

# 5. Optional: augmented_data/
Write-Host ""
Write-Host "[5/6] Checking augmented_data/..." -ForegroundColor Cyan
if (Test-Path "$baseDir\augmented_data") {
    Write-Host "  ℹ augmented_data/ exists. Delete manually if no longer needed." -ForegroundColor Yellow
    Write-Host "    (Or move to data/raw/ if keeping)" -ForegroundColor Yellow
}

# 6. Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cleanup Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Successfully deleted: $($deleted.Count) files" -ForegroundColor Green
Write-Host "Failed: $($failed.Count) files" -ForegroundColor $(if ($failed.Count -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($deleted.Count -gt 0) {
    Write-Host "Deleted files:" -ForegroundColor Green
    foreach ($file in $deleted) {
        Write-Host "  - $file" -ForegroundColor Gray
    }
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed deletions:" -ForegroundColor Red
    foreach ($file in $failed) {
        Write-Host "  - $file" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review git status: git status" -ForegroundColor White
Write-Host "  2. Stage changes: git add ." -ForegroundColor White
Write-Host "  3. Commit: git commit -m 'Reorganize directory structure'" -ForegroundColor White
Write-Host ""
Write-Host "Cleanup complete! ✅" -ForegroundColor Green
