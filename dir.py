$skip = @('__pycache__','.git','.venv','.pytest_cache','.idea','muni_lab_core.egg-info')

function Show-Tree {
  param(
    [string]$Path = ".",
    [string]$Prefix = ""
  )

  $items = Get-ChildItem -LiteralPath $Path -Directory |
           Where-Object { $skip -notcontains $_.Name } |
           Sort-Object Name

  for ($i = 0; $i -lt $items.Count; $i++) {
    $isLast = ($i -eq $items.Count - 1)

    if ($isLast) { $branch = '└── ' } else { $branch = '├── ' }
    Write-Output ($Prefix + $branch + $items[$i].Name)

    if ($isLast) { $pad = '    ' } else { $pad = '│   ' }
    Show-Tree -Path $items[$i].FullName -Prefix ($Prefix + $pad)
  }
}

Show-Tree .
