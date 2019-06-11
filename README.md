# cleverwall
An open-source neural net to detect and counter threats on a local network.

### 6/10/2019 - Little design change
After a few hours of beating my head against the wall, I've come to a conclusion about the current setup.
My current idea of how this is supposed to work out is:
- Feed a CSV file full of traffic logs (from wireshark) into the lil brain
- Iterate for eternity to find patterns in the traffic.  

Should work great, right?
Wrong. The CSV packet logs don't contain actual data. They're simply metadata.
I need to find a way to feed the *actual goddamn pcap files* into the network, then extract packet contents and analyze *that*.
I monkeyed pretty hard today. :(
