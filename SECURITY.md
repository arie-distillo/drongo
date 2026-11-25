# Security policy

drongo parses files and network packets that may come from untrusted sources (video files,
telemetry sidecars, live downlinks). We treat any crash, hang, excessive memory use or code
execution triggered by such input as a security issue.

## Reporting

Please do **not** open a public issue. Use GitHub's private vulnerability reporting
(*Security* tab &rarr; *Report a vulnerability*) or email **security@REPLACE-WITH-PROJECT-EMAIL**
with a description and, if possible, a file that reproduces the problem.

We aim to acknowledge reports within 5 working days and to publish a fix or mitigation within 30 days.

## Supported versions

Until 1.0, only the latest release receives fixes.

## Privacy

drongo makes no network connections. Telemetry contains locations; nothing leaves your process
unless your code writes it. The diagnostics report (`report.html`) loads map tiles and Leaflet from
public servers when opened in a browser, which reveals the map area to those servers.
