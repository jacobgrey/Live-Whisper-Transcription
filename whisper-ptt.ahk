#Requires AutoHotkey v2.0
#SingleInstance Force

py := A_ScriptDir "\venv\Scripts\python.exe"
client := A_ScriptDir "\src\whisper_client.py"
daemonCmd := A_ScriptDir "\scripts\start_daemon.cmd"

; Start daemon (manual hotkey)
F7::
{
    Run('cmd /c ""' daemonCmd '""')
}

; Start daemon with input device selection
+F7::
{
    Run('cmd /c ""' daemonCmd '" --select-device"')
}

F8::
{
    global py, client
    RunWait('"' py '" "' client '" START', , "Hide")
}

F8 Up::
{
    global py, client

    tmp := A_Temp "\whisper_ptt_out.txt"
    try FileDelete(tmp)

    RunWait('"' py '" "' client '" STOP --output "' tmp '"', , "Hide")

    if !FileExist(tmp) {
        ToolTip("Transcription failed: no output from daemon.`nIs the daemon running? (F7 to start)")
        SetTimer(() => ToolTip(), -4000)
        return
    }

    out := Trim(FileRead(tmp, "UTF-8"))

    if (SubStr(out, 1, 3) = "OK ")
    {
        text := SubStr(out, 4)
        if (Trim(text) = "")
        {
            ToolTip("No speech detected")
            SetTimer(() => ToolTip(), -2000)
            return
        }
        A_Clipboard := text
        Sleep 120
        Send "^v"
    }
    else if (out != "")
    {
        ToolTip("Transcription error: " SubStr(out, 1, 120))
        SetTimer(() => ToolTip(), -4000)
    }
    else
    {
        ToolTip("Transcription failed: empty response")
        SetTimer(() => ToolTip(), -4000)
    }
}

; Shutdown daemon (frees VRAM)
F9::
{
    RunWait('"' py '" "' client '" SHUTDOWN', , "Hide")
}