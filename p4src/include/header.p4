/*
 * Copyright 2019-present Open Networking Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __HEADER__
#define __HEADER__

#include "define.p4"

// Longitud del campo que contendrá la concatenacion de todas las estadisticas de un flujo. 
typedef bit<344> StatLen;


//Packet-in header. Prepended to packets sent to the controller and used to
// carry the original ingress port where the packet was received.
@controller_header("packet_in")
header packet_in_header_t {
    @field_list()
    bit<9> ingress_port;
    bit<7> _padding;
} // 2 bytes

// Packet-out header. Prepended to packets received by the controller and used
// to tell the switch on which port this packet should be forwarded.
@controller_header("packet_out")
header packet_out_header_t {
    bit<9> egress_port;
    bit<7> _padding;
} // 2 bytes


header ethernet_t {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
} // 14 bytes

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<6>  diffserv;
    bit<2>  tag;
    bit<16> len;
    bit<16> identification;
    bit<3>  flags;
    bit<13> frag_offset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdr_checksum;
    bit<32> src_addr;
    bit<32> dst_addr;
} //24 bytes

header tcp_t {
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4>  data_offset;
    bit<3>  res;
    bit<3>  ecn;
    bit<6>  ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
} // 20 bytes

header udp_t {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> len;
    bit<16> checksum;
} // 8 bytes


/* --- Orden de concatenación para cada Fx ---
bit<48> FlowDurationH;      // 6 bytes
bit<32> TotPkts_FWD;        // 4 bytes
bit<32> TotPkts_BWD;        // 4 bytes
bit<32> TotLenPkts_FWD;     // 4 bytes
bit<32> TotLenPkts_BWD;     // 4 bytes
bit<40> TotLenSquare;       // 5 bytes
bit<48> TotIAT;             // 6 bytes
bit<56> TotIATsquare ;      // 7 bytes
bit<16> WindowNum;          // 2 bytes
// 42 bytes en total (336 bits)
// -------------------------------------------
*/

header flow_t {
    bit<8>  NumFlowsByPacket;   // 1 bytes
    StatLen F1;                 // 43 bytes
}                       // Total = 44 bytes

header flow_x5_t {
    bit<8>  NumFlowsByPacket;   // 1 bytes
    StatLen F1;                 // 43 bytes
    StatLen F2;                 // 43 bytes
    StatLen F3;                 // 43 bytes
    StatLen F4;                 // 43 bytes
    StatLen F5;                 // 43 bytes
}                       // Total = 216 bytes

header flow_x10_t {
    bit<8>  NumFlowsByPacket;   // 1 bytes
    StatLen F1;                 // 43 bytes
    StatLen F2;                 // 43 bytes
    StatLen F3;                 // 43 bytes
    StatLen F4;                 // 43 bytes
    StatLen F5;                 // 43 bytes
    StatLen F6;                 // 43 bytes
    StatLen F7;                 // 43 bytes
    StatLen F8;                 // 43 bytes
    StatLen F9;                 // 43 bytes
    StatLen F10;                // 43 bytes
}                     // Total =  431 bytes


// For convenience we collect all headers under the same struct.
struct headers_t {
    ethernet_t                  ethernet;
    ipv4_t                      ipv4;
    tcp_t                       tcp;
    udp_t                       udp;
    packet_out_header_t         packet_out;
    packet_in_header_t          packet_in;
    flow_t                      flow;
    flow_x5_t                   flow_x5;
    flow_x10_t                  flow_x10;
}

// Metadata can be used to carry information from one table to another.
struct metadata_t {
    bit<32> contador;
    bit<32> index;
    bit<32> index2; //index del flujo de retorno (BWD)
    bit<16> srcP;
    bit<16> dstP;
    bit<2>  state;
    bit<2>  state2;
    bit<32> indF;
    bit<32> indB;
    //--- Estadisticas en un sentido del flujo
    bit<48> InitTimeFlowM;
    bit<48> LastTimePacketM;
    bit<32> TotPktsM;
    bit<32> TotLenPktsM;
    bit<32> PktLenMinM;
    bit<32> PktLenMaxM;
    bit<40> TotLenSquareM;

    //--- Estadisticas en el otro sentido del flujo
    bit<48> InitTimeFlowM2;
    bit<48> LastTimePacketM2;
    bit<32> TotPktsM2;
    bit<32> TotLenPktsM2;
    bit<32> PktLenMinM2;
    bit<32> PktLenMaxM2;
    bit<40> TotLenSquareM2;

    bit<16> WindowNumM;
    @field_list(1)
    bit<8> NumFlowsByPacket;
    bit<48> FlowDurationM;
    bit<48> TotIATM;
    bit<56> TotIATsquareM;
    bit<1>  tagM;

    //-------
    @field_list(1)
    StatLen Flow1;
    @field_list(1)
    StatLen Flow2;
    @field_list(1)
    StatLen Flow3;
    @field_list(1)
    StatLen Flow4;
    @field_list(1)
    StatLen Flow5;
    @field_list(1)
    StatLen Flow6;
    @field_list(1)
    StatLen Flow7;
    @field_list(1)
    StatLen Flow8;
    @field_list(1)
    StatLen Flow9;
    @field_list(1)
    StatLen Flow10;    
}

#endif
