$loky=$args[0]
if (!$loky) {
    $loky="loky"
}
$install_folder="C:\Users\$Env:UserName\AppData\Local\Temp\loky_tmp"
python -m pip install --no-user --target $install_folder $loky
Copy-Item "$install_folder\loky\*" "loky" -Recurse -Force
Remove-Item -Path "$install_folder" -Recurse

$files = $(git grep -l cloudpickle loky)
foreach ($filename in $files)
{
    Write-Host $filename
    $file = Get-ChildItem -Path $filename
    (Get-Content $file.PSPath) |
    ForEach-Object {$_ -creplace "import cloudpickle", "from joblib.externals import cloudpickle"} |
    Set-Content $file.PSPath

    (Get-Content $file.PSPath) |
    Foreach-Object { $_ -creplace "from cloudpickle import", "from joblib.externals.cloudpickle import" } |
    Set-Content $file.PSPath
}

$files = Get-ChildItem loky *.py -Recurse
foreach ($file in $files)
{
    (Get-Content $file.PSPath) | Foreach-Object { $_ -creplace "from loky", "from joblib.externals.loky" } | Set-Content $file.PSPath
}
