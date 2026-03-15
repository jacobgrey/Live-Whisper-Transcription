#Requires AutoHotkey v2.0
#SingleInstance Force

py := "D:\Programs\Whisper Speech to Text\ptt\venv\Scripts\python.exe"
client := "D:\Programs\Whisper Speech to Text\ptt\src\whisper_client.py"
daemonCmd := "D:\Programs\Whisper Speech to Text\ptt\scripts\start_daemon.cmd"

; Start daemon (manual hotkey)
F7::
{
    Run('cmd /c ""' daemonCmd '""')
}

F8::
{
    global py, client
    RunWait('cmd /c ""' py '" "' client '" START""', , "Hide")
}

F8 Up::
{
    global py, client

    tmp := A_Temp "\whisper_ptt_out.txt"
    try FileDelete(tmp)

    cmdLine := 'cmd /c ""' py '" "' client '" STOP > "' tmp '""'
    RunWait(cmdLine, , "Hide")

    if !FileExist(tmp) {
        ; Uncomment to see what happened:
        ; MsgBox "STOP produced no output file. Is the daemon running?"
        return
    }

    out := Trim(FileRead(tmp, "UTF-8"))

    if (SubStr(out, 1, 3) = "OK ")
    {
        text := SubStr(out, 4)
        A_Clipboard := text
        Sleep 120
        Send "^v"
    }
    else
    {
        ; MsgBox out
    }
}

; Shutdown daemon (frees VRAM)
F9::
{
    RunWait('"' py '" "' client '" SHUTDOWN', , "Hide")
}