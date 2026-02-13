/**
 * This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
 *
 * MIT Licence
 * Copyright (c) 2025, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

always_comb begin 
    /*
     * CONTROL SIGNALS
     * 
     * rq_(wr|rd) are two more Coyote interfaces, which act as inputs to the user application
     * They corresponds to network write/read requests, set from the host software and driver
     * Here, they are used to set Coyote's generic send queues, previously discussed in Example 7.
     */
    

    // Reads
    sq_rd.valid = rq_rd.valid;
    rq_rd.ready = sq_rd.ready;
    sq_rd.data = rq_rd.data;           // Data field holds information such as remote, virtual address, buffer length etc.
    sq_rd.data.strm = STRM_HOST;       // For RDMA, by definition data is always on the host
    sq_rd.data.dest = 1;
end

////////////////////////////////////////
// FSM for vaddr-assignment 
///////////////////////////////////////

// Array with the vaddrs for scatter operation 
logic [VADDR_BITS-1:0] vaddr_arr[4]; 

// Logic signal for indicating that all the vaddr are completely indicated by the axi_ctrl interface 
logic bench_vaddr_valid;

// Counter variable for packets to control the scattering
logic[2:0] pkt_cnt; 

// IP-core for getting the vaddr for scatter operation 
rdma_scatter_axi_ctrl_parser inst_axi_ctrl_parser (
    .aclk(aclk), 
    .aresetn(aresetn), 
    .axi_ctrl(axi_ctrl), 
    .bench_vaddr_1(vaddr_arr[0]),
    .bench_vaddr_2(vaddr_arr[1]), 
    .bench_vaddr_3(vaddr_arr[2]),
    .bench_vaddr_4(vaddr_arr[3]), 
    .bench_vaddr_valid(bench_vaddr_valid)
); 

// FSM to pass on the vaddr to the write commands and move up for scatter operations based on packets. 
always_ff @(posedge aclk) begin 
    if(!aresetn) begin 
        // Reset the packet counter
        pkt_cnt <= 3'd0;
    end else begin 
        // Increment the packet based on new RDMA WRITE-commands
        if(axis_rrsp_recv[0].tvalid && axis_rrsp_recv[0].tready && axis_rrsp_recv[0].tlast) begin 
            if(pkt_cnt == 3'd3) begin 
                pkt_cnt <= 3'd0; 
            end else begin 
                pkt_cnt <= pkt_cnt + 3'd1; 
            end 
        end
    end 
end 

// Now assign the vaddr based on the packet counter and if the vaddr are valid (set in the parser)
always_comb begin 
    if(bench_vaddr_valid) begin 
        // If the transmission is valid, assign the vaddr based on the packet counter 
        case(pkt_cnt) 
            3'd0: sq_wr.data.vaddr = vaddr_arr[0];
            3'd1: sq_wr.data.vaddr = vaddr_arr[1];
            3'd2: sq_wr.data.vaddr = vaddr_arr[2];
            3'd3: sq_wr.data.vaddr = vaddr_arr[3];
            default: sq_wr.data.vaddr = rq_wr.data.vaddr; // Default case should never happen
        endcase
    end else begin 
        // If the transmission is not yet valid, use the original vaddr 
        sq_wr.data.vaddr = rq_wr.data.vaddr;
    end

    // Pass on all data-fields except for the vaddr
    sq_wr.data.opcode   = rq_wr.data.opcode;  // Opcode (READ/WRITE)
    sq_wr.data.mode     = rq_wr.data.mode;      // Mode (e.g. for atomic operations)
    sq_wr.data.rdma     = rq_wr.data.rdma;      // RDMA (1 for RDMA operations, 0 for local operations)
    sq_wr.data.remote   = rq_wr.data.remote;  // Remote (1 for remote operations, 0 for local operations) 
    sq_wr.data.vfid     = rq_wr.data.vfid;      // VFID (for SR-IOV, not used in this example)
    sq_wr.data.pid      = rq_wr.data.pid;       // PID (process ID, for multi-process support)  
    sq_wr.data.last     = rq_wr.data.last;      // Last (indicates last beat of a packet)
    sq_wr.data.len      = rq_wr.data.len;       // Length (in bytes)
    sq_wr.data.actv     = rq_wr.data.actv;      // Active (indicates if the request is valid)
    sq_wr.data.host     = rq_wr.data.host;      // Host (1 for host memory, 0 for FPGA memory)
    sq_wr.data.offs     = rq_wr.data.offs;      // Offset (byte offset within a page) 
    sq_wr.data.rsrvd    = rq_wr.data.rsrvd;     // Reserved (for future use)

    // Write
    sq_wr.valid = rq_wr.valid;
    rq_wr.ready = sq_wr.ready;        // Data field holds information such as remote, virtual address, buffer length etc.
    sq_wr.data.strm = STRM_HOST;        // For RDMA, by definition data is always on the host
    sq_wr.data.dest = is_opcode_rd_resp(rq_wr.data.opcode) ? 0 : 1; 
end 

/*
 * DATA SIGNALS
 * 
 */
// Data streams for outgoing RDMA WRITEs (from local host to network stack to remote node)
`AXISR_ASSIGN(axis_host_recv[0], axis_rreq_send[0])

// Data streams for incoming RDMA READ RESPONSEs (from remote node to network stack to local host)
`AXISR_ASSIGN(axis_rreq_recv[0], axis_host_send[0])

// Data streams for outgoing RDMA READ RESPONSEs (from local host to network stack to remote node)
`AXISR_ASSIGN(axis_host_recv[1], axis_rrsp_send[0])

// Data streams for incoming RDMA WRITEs (from remote node to network stack to local host)
`AXISR_ASSIGN(axis_rrsp_recv[0], axis_host_send[1])

// Tie off unused interfaces
// always_comb axi_ctrl.tie_off_s();
always_comb notify.tie_off_m();
always_comb cq_rd.tie_off_s();
always_comb cq_wr.tie_off_s();

// ILA for debugging
ila_perf_rdma inst_ila_perf_rdma (
    .clk(aclk),
    .probe0(axis_host_recv[0].tvalid),      // 1
    .probe1(axis_host_recv[0].tready),      // 1
    .probe2(axis_host_recv[0].tlast),       // 1

    .probe3(axis_host_recv[1].tvalid),      // 1
    .probe4(axis_host_recv[1].tready),      // 1
    .probe5(axis_host_recv[1].tlast),       // 1

    .probe6(axis_host_send[0].tvalid),      // 1
    .probe7(axis_host_send[0].tready),      // 1
    .probe8(axis_host_send[0].tlast),       // 1

    .probe9(axis_host_send[1].tvalid),      // 1
    .probe10(axis_host_send[1].tready),     // 1
    .probe11(axis_host_send[1].tlast),      // 1

    .probe12(sq_wr.valid),                  // 1
    .probe13(sq_wr.ready),                  // 1
    .probe14(sq_wr.data),                   // 128
    .probe15(sq_rd.valid),                  // 1
    .probe16(sq_rd.ready),                  // 1
    .probe17(sq_rd.data),                   // 128

    .probe18(vaddr_arr[0]),                // 64
    .probe19(vaddr_arr[1]),                // 64
    .probe20(vaddr_arr[2]),                // 64
    .probe21(vaddr_arr[3]),                // 64
    .probe22(bench_vaddr_valid),           // 1
    .probe23(pkt_cnt),                     // 3

    .probe24(axi_ctrl.wvalid), 
    .probe25(axi_ctrl.awvalid), 
    .probe26(axi_ctrl.awaddr),              // 64
    .probe27(axi_ctrl.wdata)                // 64   
);
