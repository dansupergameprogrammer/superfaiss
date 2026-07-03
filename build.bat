@echo off
rem Quick MSVC build + test. For other compilers use CMake.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
	echo VsDevCmd.bat not found at %VSDEVCMD%
	exit /b 1
)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0
if not exist out mkdir out
cl /nologo /std:c++17 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\kernels.cpp src\kernels_avx2.cpp src\alloc.cpp src\validate.cpp src\bake.cpp src\query.cpp ^
	tests\test_main.cpp /Fo:out\ /Fe:out\superfaiss_tests.exe
if errorlevel 1 exit /b 1
out\superfaiss_tests.exe
