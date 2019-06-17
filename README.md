cleverwall
==========
Cleverwall is an open-source neural network to detect and counter threats on a local network.

How does it work?
-----------------
1. Data in the form of a packet capture is compared against snort and rules are generated.
2. Packet capture are loaded into cleverwall.
3. `cleverwall` will use some form of neural net (feed-forward is the current objective) to "learn" patterns in packet data (hex dumps) that may be dangerous.
4. Since cleverwall will work as an IDS/IPS, it will either directly drop packets that follow the patterns it learns or it will simply generate more rules for an existing one like Snort or Suricata.
