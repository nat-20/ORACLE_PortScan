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
	if len(sys.argv) != 2:
		print("Parameters: --IP_dst")
		sys.exit(1)

	DstMac = RandMAC()	
	#DstIP = sys.argv[1]
	#DstIP = "10.0.0.1"
	SrcIP = "10.0.0.1"
	#SrcIP = sys.argv[1]
	SrcPort = 5232
	DstPort = 80
	conf.checkIPaddr = False

	#Dests = ["10.0.0.100","10.0.0.20","10.0.0.30","10.0.0.40","10.0.0.50","10.0.0.60","10.0.0.70","10.0.0.80","10.0.0.90","10.0.0.10"]
	Dests = ["10.0.0.10"]
	#Dests = ["10.0.0.100","10.0.0.20","10.0.0.30","10.0.0.40","10.0.0.50","10.0.0.60","10.0.0.70","10.0.0.80","10.0.0.90","10.0.0.10",
	#		"10.0.0.200","10.0.0.250","10.0.100.1","10.0.100.50","10.0.100.100","10.0.100.200","10.0.100.250","10.0.200.50"]#,"10.0.200.150","10.0.200.200"]	

	for DstIP in Dests:
		#/--------------------------------------- HandShake ---------------------------------------------------------------
		Packet = Ether(dst=DstMac)/IP(dst=DstIP,src=SrcIP)/TCP(sport=SrcPort, dport=DstPort, flags=0x02) #SYN
		sendpacket(Packet,1,"TCP-FWD-HandShake / SYN")
		#input("Pulsa una tecla para continuar...") 
		Packet = Ether(dst=DstMac)/IP(dst=SrcIP,src=DstIP)/TCP(sport=DstPort, dport=SrcPort, flags=0x12) #SYN-ACK
		sendpacket(Packet,1,"TCP-BWD-HandShake / SYN-ACK")
		#input("Pulsa una tecla para continuar...") 
		Packet = Ether(dst=DstMac)/IP(dst=DstIP,src=SrcIP)/TCP(sport=SrcPort, dport=DstPort, flags=0x10) #ACK
		sendpacket(Packet,1,"TCP-FWD-HandShake / ACK")

		#input("Pulsa una tecla para continuar...") 

		#/--------------------------------------- Dada Transfer -----------------------------------------------------------
		Packet = Ether(dst=DstMac)/IP(dst=DstIP,src=SrcIP,tos=3)/TCP(sport=SrcPort, dport=DstPort, flags=0x10)/Raw(load="hola") #ACK
		sendpacket(Packet,1,"TCP-FWD-Data / ACK")
		Packet = Ether(dst=DstMac)/IP(dst=DstIP,src=SrcIP)/TCP(sport=SrcPort, dport=DstPort, flags=0x04)/Raw(load="hola2") #RST
		sendpacket(Packet,1,"TCP-FWD-data / RST")
		#input("Pulsa una tecla para continuar...") 
		Packet = Ether(dst=DstMac)/IP(dst=SrcIP,src=DstIP)/TCP(sport=DstPort, dport=SrcPort, flags=0x10)/Raw(load="hola parce") #ACK
		sendpacket(Packet,1,"TCP-BWD-Data / ACK")
		Packet = Ether(dst=DstMac)/IP(dst=SrcIP,src=DstIP)/TCP(sport=DstPort, dport=SrcPort, flags=0x08)/Raw(load="hola parce 2") #PSH
		sendpacket(Packet,1,"TCP-BWD-data / PSH")

		#input("Pulsa una tecla para continuar...") 

		#/--------------------------------------- End Connection ----------------------------------------------------------
		Packet = Ether(dst=DstMac)/IP(dst=DstIP,src=SrcIP)/TCP(sport=SrcPort, dport=DstPort, flags=0x11) #FIN-ACK
		sendpacket(Packet,1,"TCP-FWD-Fin_Connection / FIN_ACK")
		#input("Pulsa una tecla para continuar...") 
		Packet = Ether(dst=DstMac)/IP(dst=SrcIP,src=DstIP)/TCP(sport=DstPort, dport=SrcPort, flags=0x11) #FIN-ACK
		sendpacket(Packet,1,"TCP-BWD-Fin_Connection / FIN_ACK")

		


		"""Packet = Ether(dst=DstMac)/IP(dst=DstIP,src=SrcIP)/UDP(sport=SrcPort, dport=DstPort) 
		sendpacket(Packet,1,"UDP-FWD-Data")
		#input("Pulsa una tecla para continuar...")
		Packet = Ether(dst=DstMac)/IP(dst=DstIP,src=SrcIP)/UDP(sport=SrcPort, dport=DstPort)/"prueba FWD" 
		sendpacket(Packet,1,"UDP-FWD-Data")
		#input("Pulsa una tecla para continuar...") 
		Packet = Ether(dst=DstMac)/IP(dst=SrcIP,src=DstIP)/UDP(sport=DstPort, dport=SrcPort) 
		sendpacket(Packet,1,"UDP-BWD-Data")
		#input("Pulsa una tecla para continuar...") 
		Packet = Ether(dst=DstMac)/IP(dst=SrcIP,src=DstIP)/UDP(sport=DstPort, dport=SrcPort)/"prueba BWD" 
		sendpacket(Packet,1,"UDP-BWD-Data")"""

	#input("Pulsa una tecla para continuar...") 

	#/--------------------------------------- New Connection ----------------------------------------------------------
	#Packet = Ether(dst=DstMac)/IP(dst=DstIP,src=SrcIP)/TCP(sport=SrcPort, dport=DstPort, flags=0x02) #SYN
	#sendpacket(Packet,2*len(Dests),"TCP-FWD-HandShake / SYN")
	
	

