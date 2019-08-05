<p align="center"><img src="assets/logo.png" width=100/></p>
<h1 align="center">net-csi</h1>
<p>an open-source neural net utility designed and optimized for detecting potential threats in network traffic captures.</p>

<hr>

An important note: THIS PROJECT IS NOT YET COMPLETE, as I basically have no clue what I'm doing. It may be another couple of months. Here are my current plans:

Installation
------------
I'll establish a clear way of doing this once I finalize net-csi.

Progress
--------
- [x] A traffic capture (Wireshark PCAP) is analyzed with `suricata` and a base ruleset is generated.
- [x] The original capture data and `suricata`'s resulting `eve.json` log are loaded into net-csi as a `TrafficDataset`.
- [x] `net-csi` will use some model of neural net (The current model is a residual neural network) to identify patterns in packet data (hex dumps) that correspond to malicious activity.
- [ ] `net-csi` will then flag packets it deems malicious and log them, much like `suricata`.