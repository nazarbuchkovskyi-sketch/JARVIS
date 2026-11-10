' Starts Jarvis without a console window: only the Jarvis window appears.
' Double-click this file (or the "Jarvis" desktop shortcut). Close the Jarvis window to stop him.
' Logs: logs\jarvis.log. The first run (no UI build or models yet) uses the visible start.bat instead.
Option Explicit
Dim sh, fso, root, rc, http
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = root

' Already running: just bring up its window.
On Error Resume Next
Set http = CreateObject("MSXML2.XMLHTTP")
http.Open "GET", "http://127.0.0.1:8765/", False
http.Send
If Err.Number = 0 And http.Status = 200 Then
  sh.Run "msedge --app=http://127.0.0.1:8765/ --user-data-dir=""" & root & "\.window-profile""", 1, False
  WScript.Quit 0
End If
Err.Clear
On Error GoTo 0

' First run: downloads and builds need a visible console.
If Not fso.FileExists(root & "\ui\dist\index.html") Or Not fso.FileExists(root & "\models\.ready") Then
  sh.Run "cmd /c """"" & root & "\start.bat""""", 1, False
  WScript.Quit 0
End If

rc = sh.Run("python -c ""import fastapi, onnx_asr, piper, silero_vad, sounddevice, cv2""", 0, True)
If rc <> 0 Then
  MsgBox "Python packages are missing. Run setup.bat first.", vbExclamation, "Jarvis"
  WScript.Quit 1
End If

' 0 = hidden window; Jarvis opens its own Edge window when ready.
sh.Run "python -m jarvis", 0, False
