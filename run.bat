@echo off
set PATH=C:\c++\libs-v140\apache-arrow\bin;C:\c++\libs-v140\xlnt\bin;%~dp0\native_libs\src\x64\Debug;%PATH%
set LUNA_LIBS_PATH=F:\dev\luna\stdlib
echo Call luna run --target %~dp0
stack --stack-yaml F:\dev\luna\stack.yaml exec -- cmd.exe