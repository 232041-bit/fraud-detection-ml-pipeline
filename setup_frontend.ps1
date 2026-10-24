# Run from project root:  .\setup_frontend.ps1
# Generates the Angular app (Manual Stage 8) and applies the ready-made fraud form (Stage 9).
$ErrorActionPreference = "Stop"
if (Test-Path frontend) { Write-Host "frontend\ already exists - delete it first if you want to regenerate."; exit 1 }

ng.cmd new frontend --routing --style=css --skip-git --defaults
if ($LASTEXITCODE -ne 0) { throw "ng new failed" }

Copy-Item -Path frontend_overlay\* -Destination frontend -Recurse -Force
if (Test-Path frontend\src\app\app.spec.ts) { Remove-Item frontend\src\app\app.spec.ts }

# Manual 9.3: add provideHttpClient() to app.config.ts (keeps generated providers)
$cfg = "frontend\src\app\app.config.ts"
$t = Get-Content $cfg -Raw
if ($t -notmatch "provideHttpClient") {
  $t = "import { provideHttpClient } from '@angular/common/http';`n" + $t
  $t = $t -replace "providers:\s*\[", "providers: [`n    provideHttpClient(),"
  Set-Content -Path $cfg -Value $t -Encoding UTF8
}
Write-Host "Frontend ready. Next: cd frontend ; npm run build"
