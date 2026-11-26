#define NOMINMAX
#include <windows.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::wstring quoteArgument(const std::wstring& value) {
    std::wstring quoted = L"\"";
    unsigned int slashes = 0;
    for (const wchar_t character : value) {
        if (character == L'\\') {
            ++slashes;
        } else if (character == L'"') {
            quoted.append(slashes * 2 + 1, L'\\');
            quoted.push_back(L'"');
            slashes = 0;
        } else {
            quoted.append(slashes, L'\\');
            slashes = 0;
            quoted.push_back(character);
        }
    }
    quoted.append(slashes * 2, L'\\');
    quoted.push_back(L'"');
    return quoted;
}

std::filesystem::path findProjectRoot(const std::filesystem::path& executable) {
    for (auto candidate = executable.parent_path(); !candidate.empty(); candidate = candidate.parent_path()) {
        if (std::filesystem::exists(candidate / L"python" / L"expense_app.py")) return candidate;
        if (candidate == candidate.root_path()) break;
    }
    return {};
}
}

int wmain(int argc, wchar_t* argv[]) {
    const bool useGui = argc == 1 || (argc >= 2 && std::wstring(argv[1]) == L"--gui");
    if (argc > 3 || (!useGui && argc < 2)) {
        std::wcerr << L"Uso: expense_reader [--gui [gastos.csv]] | <imagen|documento.pdf> [gastos.csv] | --camera [gastos.csv] (sin argumentos abre el menú)\n";
        return 2;
    }

    const bool useCamera = !useGui && std::wstring(argv[1]) == L"--camera";
    const std::filesystem::path image = (useGui || useCamera) ? std::filesystem::path{}
                                                   : std::filesystem::absolute(argv[1]);
    if (!useGui && !useCamera && !std::filesystem::is_regular_file(image)) {
        std::wcerr << L"No se encontró el archivo indicado.\n";
        return 2;
    }

    std::wstring modulePath(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, modulePath.data(), static_cast<DWORD>(modulePath.size()));
    if (length == 0 || length >= modulePath.size()) {
        std::wcerr << L"No se pudo resolver la ruta del ejecutable.\n";
        return 1;
    }
    modulePath.resize(length);
    const auto projectRoot = findProjectRoot(modulePath);
    if (projectRoot.empty()) {
        std::wcerr << L"No se encontró python/expense_app.py junto al proyecto.\n";
        return 1;
    }

    const auto script = projectRoot / L"python" / L"expense_app.py";
    const auto pythonInVenv = projectRoot / L".venv" / L"Scripts" / L"python.exe";
    std::filesystem::path python = pythonInVenv;
    if (useGui) {
        const auto pythonwInVenv = projectRoot / L".venv" / L"Scripts" / L"pythonw.exe";
        if (std::filesystem::exists(pythonwInVenv)) python = pythonwInVenv;
    }
    if (!std::filesystem::exists(python)) {
        std::wstring foundPath(32768, L'\0');
        const DWORD foundLength = SearchPathW(nullptr, L"python.exe", nullptr,
                                               static_cast<DWORD>(foundPath.size()), foundPath.data(), nullptr);
        if (foundLength == 0 || foundLength >= foundPath.size()) {
            std::wcerr << L"No se encontró Python. Instálalo en .venv o agrégalo al PATH.\n";
            return 1;
        }
        foundPath.resize(foundLength);
        python = foundPath;
    }

    std::wstring command = quoteArgument(python.wstring()) + L" " + quoteArgument(script.wstring()) + L" ";
    command += useGui ? L"--gui" : useCamera ? L"--camera" : quoteArgument(image.wstring());
    if (argc == 3) {
        command += L" --csv " + quoteArgument(std::filesystem::absolute(argv[2]).wstring());
    }
    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    if (!useGui) {
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    }
    PROCESS_INFORMATION process{};
    const BOOL inheritHandles = useGui ? FALSE : TRUE;
    const DWORD creationFlags = useGui ? CREATE_NO_WINDOW : 0;
    if (!CreateProcessW(python.c_str(), mutableCommand.data(), nullptr, nullptr, inheritHandles, creationFlags, nullptr,
                        projectRoot.c_str(), &startup, &process)) {
        std::wcerr << L"No se pudo iniciar Python. Código de Windows: " << GetLastError() << L"\n";
        return 1;
    }
    CloseHandle(process.hThread);
    if (useGui) {
        CloseHandle(process.hProcess);
        return 0;
    }

    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exitCode = 1;
    GetExitCodeProcess(process.hProcess, &exitCode);
    CloseHandle(process.hProcess);
    return static_cast<int>(exitCode);
}
