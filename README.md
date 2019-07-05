cleverwall
==========
Cleverwall is an open-source neural network to detect and counter threats on a local network.

How does it work?
-----------------
1. Data in the form of a packet capture is analyzed with `suricata` and a base ruleset is generated.
2. Capture data PCAP and `suricata`'s log PCAP are loaded into cleverwall as a `TrafficDataset`.
3. `cleverwall` will use some model of neural net (feed-forward is the current objective, convolutional may come later) to identify patterns in packet data (hex dumps) that correspond to malicious activity.
4. Since cleverwall will work as an IDS/IPS, it will either directly drop packets that follow the patterns it learns or it will simply generate more rules for an existing one (in our case, `suricata`).
