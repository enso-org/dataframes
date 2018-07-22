@echo off
set PATH=C:\c++\libs-v140\apache-arrow\bin;C:\c++\libs-v140\xlnt\bin\x64;%~dp0\native_libs\src\x64\Debug;%PATH%
set LUNA_LIBS_PATH=F:\dev\luna-core\stdlib
echo Call luna run --target %~dp0
stack --stack-yaml F:\dev\luna-core\stack.yaml exec -- cmd.exe