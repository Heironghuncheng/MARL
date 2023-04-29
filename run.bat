set b=%cd%
start %b%/bats/main.bat
start %b%/bats/alarmer.bat
start %b%/bats/tensorboard.bat
ping -n 5 127.0.0.1>nul
start msedge http://localhost:6006/