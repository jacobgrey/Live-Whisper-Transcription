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

; Open Explorer to the Transcribe Drop script (drag files onto it)
+F8::
{
    dropScript := A_ScriptDir "\scripts\Transcribe Drop.cmd"
    Run('explorer.exe /select,"' dropScript '"')
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
        A_Clipboard := ""
        A_Clipboard := text
        if !ClipWait(2) {
            ToolTip("Clipboard failed — text: " SubStr(text, 1, 40))
            SetTimer(() => ToolTip(), -4000)
            return
        }
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

; Quit this AHK script
+F9::
{
    ExitApp()
}

; Open a terminal at the project folder
F6::
{
    try Run('wt.exe -d "' A_ScriptDir '"')
    catch Run('cmd.exe /K cd /d "' A_ScriptDir '"')
}

; Toggle suspend (disables/re-enables every hotkey above except these two)
#SuspendExempt On
^+F9::Suspend()
^+F7::Suspend()
#SuspendExempt Off