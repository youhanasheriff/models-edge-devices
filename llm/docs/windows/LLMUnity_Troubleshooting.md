# LLMUnity Troubleshooting Guide (Windows)

## Error: "LLM service couldn't be created"

When loading a model in Unity with LLMUnity, you may encounter this error:

```
Tried architecture: avx512, error: LLM 1: Error loading the model!
Tried architecture: avx2, error: LLM 1: Error loading the model!
Tried architecture: avx, error: LLM 1: Error loading the model!
Tried architecture: noavx, error: LLM 1: Error loading the model!
LLM service couldn't be created
```

> **Note:** If the error occurs on ALL architectures (including noavx), this is a Windows environment issue, not a model compatibility problem.

---

## Root Causes

### 1. Special Characters in File Path (Most Common)

Based on [LLMUnity Issue #349](https://github.com/undreamai/LLMUnity/issues/349), paths containing Unicode, spaces, or special characters cause loading failures.

**Examples of problematic paths:**
- `C:\Users\José\Documents\...` (accented characters)
- `C:\Masaüstü\...` (Turkish characters)
- `C:\My Projects\Unity Game\...` (spaces)

### 2. Antivirus/Firewall Blocking

Windows Defender or third-party antivirus may block `undreamai_server.exe` or `archchecker.dll`.

### 3. Missing Visual C++ Redistributable

LLMUnity requires Visual C++ Redistributable 2015-2022 for native DLL loading.

---

## Fixes

### Fix 1: Move Files to ASCII-Only Path ✅

**Recommended first step:**

1. Create a simple folder:
   ```
   C:\LLM\models\
   ```

2. Move your GGUF model to this folder

3. Move your Unity project to:
   ```
   C:\Projects\YourGame\
   ```

4. Update the model path in Unity's LLMUnity component

---

### Fix 2: Whitelist in Antivirus

1. Open **Windows Security** → **Virus & threat protection**
2. Go to **Virus & threat protection settings** → **Manage settings**
3. Under **Exclusions**, click **Add or remove exclusions**
4. Add exclusions for:
   - Your Unity project folder
   - LLMUnity StreamingAssets folder
   - `%LOCALAPPDATA%\Temp\` (LLMUnity temp files)

---

### Fix 3: Install Visual C++ Redistributable

1. Download: [VC++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Also install the x86 version
3. Restart your computer
4. Relaunch Unity

---

### Fix 4: Configure Windows Firewall

1. Open **Windows Defender Firewall**
2. Click **Allow an app through firewall**
3. Click **Change settings** → **Allow another app**
4. Add `undreamai_server.exe` from StreamingAssets
5. Enable both **Private** and **Public** networks

---

### Fix 5: Kill Conflicting Processes

1. Open **Task Manager** (Ctrl+Shift+Esc)
2. Look for running `llamafile` or `undreamai` processes
3. End those processes
4. Restart Unity

---

## Verification

After applying fixes:

1. Restart Unity completely
2. Optionally clear the Library folder for a fresh rebuild
3. Re-add the LLMUnity component
4. Select a built-in model first to test
5. Check the console for errors

---

## Quick Reference

| Fix | When to Use |
|-----|-------------|
| ASCII path | Path contains Unicode/spaces/special chars |
| Antivirus whitelist | Security software blocking files |
| VC++ Redistributable | Missing DLL errors |
| Firewall exception | Network connection issues |
| Kill processes | Port conflict errors |

---

## References

- [LLMUnity GitHub Issues](https://github.com/undreamai/LLMUnity/issues)
- [Issue #349: Special characters in path](https://github.com/undreamai/LLMUnity/issues/349)
- [Issue #358: Architecture compatibility](https://github.com/undreamai/LLMUnity/issues/358)
