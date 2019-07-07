$(sudo suricata -r $1 -l . && rm -f *.log log.pcap.* && sudo chown $USERNAME eve.json) > /dev/null
