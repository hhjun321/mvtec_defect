# P1 데이터 준비 전체 실행 — SOP-MVTEC-CABLE-001 §4
# 사용: powershell -ExecutionPolicy Bypass -File src\data\run_p1.ps1
$ErrorActionPreference = "Stop"
$py = "D:\project\mvtec_defect\.venv\Scripts\python.exe"
$src = "D:\project\mvtec_defect\src\data"

Write-Host "=== SOP-DATA-01 무결성 검증 ===" -ForegroundColor Cyan
& $py "$src\verify_dataset.py"; if ($LASTEXITCODE -ne 0) { throw "SOP-DATA-01 FAIL" }

Write-Host "`n=== SOP-DATA-02 라벨 변환 ===" -ForegroundColor Cyan
& $py "$src\masks_to_yolo.py"; if ($LASTEXITCODE -ne 0) { throw "SOP-DATA-02 FAIL" }

Write-Host "`n=== SOP-DATA-03 분할 생성 ===" -ForegroundColor Cyan
& $py "$src\make_splits.py"; if ($LASTEXITCODE -ne 0) { throw "SOP-DATA-03 FAIL" }

Write-Host "`n=== 스모크 체크 ===" -ForegroundColor Cyan
& $py "$src\smoke_check.py"; if ($LASTEXITCODE -ne 0) { throw "SMOKE FAIL" }

Write-Host "`nP1 완료. reports\label_check\ 육안 확인 후 P2 착수." -ForegroundColor Green
