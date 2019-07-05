$(sudo suricata -r $1 -l . && rm -f *.log *.json && sudo chown $USERNAME log.pcap.*) > /dev/null
