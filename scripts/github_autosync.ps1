param(
    [string]$RepoPath = ".",
    [string]$Branch = "master",
    [int]$IntervalSeconds = 60,
    [switch]$RunOnce
)

$ErrorActionPreference = "Stop"

function Invoke-Sync {
    param([string]$Path, [string]$TargetBranch)

    Push-Location $Path
    try {
        $inside = git rev-parse --is-inside-work-tree 2>$null
        if ($inside -ne "true") {
            throw "Not a git repository: $Path"
        }

        $remote = git remote get-url origin 2>$null
        if (-not $remote) {
            throw "Remote 'origin' is not configured. Run: git remote add origin <your-github-repo-url>"
        }

        git add -A
        $status = git status --porcelain
        if (-not $status) {
            Write-Host "$(Get-Date -Format s) no changes"
            return
        }

        $msg = "auto-sync $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        git commit -m $msg | Out-Host
        git push origin $TargetBranch | Out-Host
        Write-Host "$(Get-Date -Format s) synced to $TargetBranch"
    }
    finally {
        Pop-Location
    }
}

if ($RunOnce) {
    Invoke-Sync -Path $RepoPath -TargetBranch $Branch
    exit 0
}

while ($true) {
    try {
        Invoke-Sync -Path $RepoPath -TargetBranch $Branch
    }
    catch {
        Write-Host "$(Get-Date -Format s) sync failed: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
