/**
 * This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
 *
 * MIT Licence
 * Copyright (c) 2025-2026, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

localparam integer INPUT_DATA_BITWIDTH = 64;

// Input data width conversion - Coyote uses 512-bit AXI Streams but the 3D mutual information (MI) kernel expects 64-bit inputs
logic [N_STRM_AXI-1:0][INPUT_DATA_BITWIDTH-1:0]     axis_host_recv_tdata;
logic [N_STRM_AXI-1:0][INPUT_DATA_BITWIDTH/8-1:0]   axis_host_recv_tkeep;
logic [N_STRM_AXI-1:0][INPUT_DATA_BITWIDTH-1:0]     axis_host_recv_tid;
logic [N_STRM_AXI-1:0]                              axis_host_recv_tlast;
logic [N_STRM_AXI-1:0]                              axis_host_recv_tvalid;
logic [N_STRM_AXI-1:0]                              axis_host_recv_tready;

for (genvar i = 0; i < N_STRM_AXI; i++) begin
    dwidth_input_512_64 inst_dwidth_input (
        .aclk(aclk),
        .aresetn(aresetn),

        .s_axis_tdata(axis_host_recv[i].tdata),
        .s_axis_tvalid(axis_host_recv[i].tvalid),
        .s_axis_tready(axis_host_recv[i].tready),
        .s_axis_tkeep(axis_host_recv[i].tkeep),
        .s_axis_tlast(axis_host_recv[i].tlast),
        .s_axis_tid(i),

        .m_axis_tdata(axis_host_recv_tdata[i]),
        .m_axis_tvalid(axis_host_recv_tvalid[i]),
        .m_axis_tready(axis_host_recv_tready[i]),
        .m_axis_tkeep(axis_host_recv_tkeep[i]),
        .m_axis_tlast(axis_host_recv_tlast[i]),
        .m_axis_tid(axis_host_recv_tid[i])
    );
end

// Output data width conversion - the 3D MI kernel produces 64-bit outputs but Coyote expects 512-bit AXI Streams, so we need to widen the output stream
logic [INPUT_DATA_BITWIDTH-1:0]     axis_host_send_tdata;
logic [INPUT_DATA_BITWIDTH/8-1:0]   axis_host_send_tkeep;
logic                               axis_host_send_tlast;
logic                               axis_host_send_tvalid;
logic                               axis_host_send_tready;

dwidth_output_32_512 inst_dwidth_output (
    .aclk(aclk),
    .aresetn(aresetn),

    .s_axis_tdata(axis_host_send_tdata),
    .s_axis_tvalid(axis_host_send_tvalid),
    .s_axis_tready(axis_host_send_tready),
    .s_axis_tkeep(axis_host_send_tkeep),
    .s_axis_tlast(axis_host_send_tlast),
    .s_axis_tid(0),

    .m_axis_tdata(axis_host_send[0].tdata),
    .m_axis_tvalid(axis_host_send[0].tvalid),
    .m_axis_tready(axis_host_send[0].tready),
    .m_axis_tkeep(axis_host_send[0].tkeep),
    .m_axis_tlast(axis_host_send[0].tlast),
    .m_axis_tid(0)
);

// 3D Mutual Information Kernel
mutual_information_master_hls_ip inst_mutual_information_master (
    .ap_clk                     (aclk),
    .ap_rst_n                   (aresetn),

    .s_input_img_TDATA          (axis_host_recv_tdata[0]),
    .s_input_img_TKEEP          (axis_host_recv_tkeep[0]),
    .s_input_img_TLAST          (axis_host_recv_tlast[0]),
    .s_input_img_TSTRB          (0),
    .s_input_img_TVALID         (axis_host_recv_tvalid[0]),
    .s_input_img_TREADY         (axis_host_recv_tready[0]),

    .s_input_ref_TDATA          (axis_host_recv_tdata[1]),
    .s_input_ref_TKEEP          (axis_host_recv_tkeep[1]),
    .s_input_ref_TLAST          (axis_host_recv_tlast[1]),
    .s_input_ref_TSTRB          (0),
    .s_input_ref_TVALID         (axis_host_recv_tvalid[1]),
    .s_input_ref_TREADY         (axis_host_recv_tready[1]),

    .s_n_couples_TDATA          (axis_host_recv_tdata[2]),
    .s_n_couples_TKEEP          (axis_host_recv_tkeep[2]),
    .s_n_couples_TLAST          (axis_host_recv_tlast[2]),
    .s_n_couples_TSTRB          (0),
    .s_n_couples_TVALID         (axis_host_recv_tvalid[2]),
    .s_n_couples_TREADY         (axis_host_recv_tready[2]),

    .s_mutual_info_TDATA        (axis_host_send_tdata),
    .s_mutual_info_TKEEP        (axis_host_send_tkeep),
    .s_mutual_info_TLAST        (axis_host_send_tlast),
    .s_mutual_info_TSTRB        (),
    .s_mutual_info_TVALID       (axis_host_send_tvalid),
    .s_mutual_info_TREADY       (axis_host_send_tready),
    
    .s_axi_control_ARADDR       (axi_ctrl.araddr),
    .s_axi_control_ARVALID      (axi_ctrl.arvalid),
    .s_axi_control_ARREADY      (axi_ctrl.arready),
    .s_axi_control_AWADDR       (axi_ctrl.awaddr),
    .s_axi_control_AWVALID      (axi_ctrl.awvalid),
    .s_axi_control_AWREADY      (axi_ctrl.awready),
    .s_axi_control_RDATA        (axi_ctrl.rdata),
    .s_axi_control_RRESP        (axi_ctrl.rresp),
    .s_axi_control_RVALID       (axi_ctrl.rvalid),
    .s_axi_control_RREADY       (axi_ctrl.rready),
    .s_axi_control_WDATA        (axi_ctrl.wdata),
    .s_axi_control_WSTRB        (axi_ctrl.wstrb),
    .s_axi_control_WVALID       (axi_ctrl.wvalid),
    .s_axi_control_WREADY       (axi_ctrl.wready),
    .s_axi_control_BRESP        (axi_ctrl.bresp),
    .s_axi_control_BVALID       (axi_ctrl.bvalid),
    .s_axi_control_BREADY       (axi_ctrl.bready)
);

// Tie-off unused signals to avoid synthesis problems
always_comb sq_rd.tie_off_m();
always_comb sq_wr.tie_off_m();
always_comb cq_rd.tie_off_s();
always_comb cq_wr.tie_off_s();
always_comb notify.tie_off_m();
always_comb axis_host_send[1].tie_off_m();
always_comb axis_host_send[2].tie_off_m();
