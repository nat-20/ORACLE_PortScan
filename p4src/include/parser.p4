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

#ifndef __PARSER__
#define __PARSER__

#include "define.p4"


parser c_parser(packet_in packet,
                  out headers_t hdr,
                  inout metadata_t meta,
                  inout standard_metadata_t standard_metadata) {

    // A P4 parser is described as a state machine, with initial state "start"
    // and final one "accept". Each intermediate state can specify the next
    // state by using a select statement over the header fields extracted.
    state start {
        transition select(standard_metadata.ingress_port) {
            CPU_PORT: parse_packet_out;
            default: parse_ethernet;
        }
    }

    state parse_packet_out {
        packet.extract(hdr.packet_out);
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETH_TYPE_IPV4   : parse_ipv4;
            default         : accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol){
            PROTO_TCP   : parse_tcp;
            PROTO_UDP   : parse_udp;
            default     : accept;
        }
    }

    state parse_udp {
        packet.extract(hdr.udp);
        meta.srcP = hdr.udp.src_port;
        meta.dstP = hdr.udp.dst_port;
        transition accept;
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        meta.srcP = hdr.tcp.src_port;
        meta.dstP = hdr.tcp.dst_port;
        transition accept;
    }

}

//------------------------------------------------------------------------------
// DEPARSER
//------------------------------------------------------------------------------

control c_deparser(packet_out packet, in headers_t hdr) {
    apply {
        // Emit headers on the wire in the following order.
        // Only valid headers are emitted.
        packet.emit(hdr.packet_in);
        packet.emit(hdr.ethernet);
        packet.emit(hdr.flow);
        packet.emit(hdr.flow_x5);
        packet.emit(hdr.flow_x10);
        packet.emit(hdr.ipv4);        
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

#endif
