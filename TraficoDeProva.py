#! /usr/bin/python
#
# run it this way: python3 attack.py <IP_Target>
#


from scapy.all import *
from os import popen
import threading
import logging
import sys

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] (%(threadname)-s) %(message)s')


def sendpacket(Packet,num,proto):
	interface = popen('ifconfig | awk \'/eth0/ {print$1}\'').read()
	print('phisic port:',interface) #rstrip function removes any trailing characters.

	mac = RandMAC()
	for x in range(0,int(num)):
		Packet['Ether'].src = mac 
		sendp(Packet,iface=interface.rstrip(),verbose=False)
		print("packet #", proto, x+1)


if __name__ == '__main__':
	if len(sys.argv) != 4:
		print("Parameters: --IP_dst, --Protocol(ICMP,UDP,TCP), --#packets")
		sys.exit(1)

	DstMac = RandMAC()	
	DstIP = sys.argv[1]
	conf.checkIPaddr = False

	if sys.argv[2]=="ICMP":	
		Packet = Ether(dst=DstMac)/IP(dst=DstIP)/ICMP()
		sendpacket(Packet,sys.argv[3],sys.argv[2])
	if sys.argv[2]=="UDP":
		Packet = Ether(dst=DstMac)/IP(dst=DstIP)/UDP(dport=5569,sport=6666)/Raw(load="payload ready")
		sendpacket(Packet,sys.argv[3],sys.argv[2])
	if sys.argv[2]=="TCP":
		Packet = Ether(dst=DstMac)/IP(dst=DstIP,tos=3)/TCP(dport=5569,sport=6666)/Raw(load="payload ready")
		sendpacket(Packet,sys.argv[3],sys.argv[2])


	"""Packet = Ether(dst=DstMac)/IP(dst="10.0.0.200")/TCP(dport=5569,sport=6666)/Raw(load="sebastian gomez")
	sendpacket(Packet,1,"TCP")	

	input("Pulsa una tecla para continuar...")

	Packet = Ether(dst=DstMac)/IP(dst=DstIP)/UDP(dport=5569,sport=6666)/Raw(load="payload ready mada fuck")
	sendpacket(Packet,sys.argv[3],sys.argv[2])

	input("Pulsa una tecla para continuar...")

	Packet = Ether(dst=DstMac)/IP(dst=DstIP)/ICMP()
	sendpacket(Packet,sys.argv[3],sys.argv[2])

	input("Pulsa una tecla para continuar...")

	Packet = Ether(dst=DstMac)/IP(dst=DstIP)/ICMP()
	sendpacket(Packet,sys.argv[3],sys.argv[2])"""
	


	"""Packet = Ether(dst=DstMac)/IP(dst=DstIP)/TCP(dport=5569,sport=6666)/Raw(load="payload ready")
	while 1==1:
		sendpacket(Packet,sys.argv[3],sys.argv[2])"""