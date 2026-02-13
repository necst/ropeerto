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

import lynxTypes::*;

/**
 * perf_fpga_axi_ctrl_parser
 * @brief Reads from/wites to the AXI Lite stream containing the benchmark data
 * 
 * @param[in] aclk Clock signal
 * @param[in] aresetn Active low reset signal
 
 * @param[in/out] axi_ctrl AXI Lite Control signal, from/to the host via PCIe and XDMA

 * @param[out] bench_ctrl Benchmark trigger to start reads/writes
 * @param[in] bench_done Number of completed reps
 * @param[in] bench_timer Benchmark timer
 * @param[out] bench_vaddr Buffer virtual address for reading/writing
 * @param[out] bench_len Buffer length (size in bytes) for reading/writing
 * @param[out] bench_pid Coyote thread ID
 * @param[out] bench_n_reps Requested number (from the user software) of read/write reps
 * @param[out] bench_n_beats Number of AXI data beats (check vfpga_top.svh and README for description)
 */
module rdma_scatter_axi_ctrl_parser (
  input  logic                        aclk,
  input  logic                        aresetn,
  
  AXI4L.s                             axi_ctrl,

  output logic [VADDR_BITS-1:0]        bench_vaddr_1,
  output logic [VADDR_BITS-1:0]        bench_vaddr_2,
  output logic [VADDR_BITS-1:0]        bench_vaddr_3,
  output logic [VADDR_BITS-1:0]        bench_vaddr_4,
  output logic                         bench_vaddr_valid
);

/////////////////////////////////////
//          CONSTANTS             //
///////////////////////////////////
localparam integer N_REGS = 5;
localparam integer ADDR_MSB = $clog2(N_REGS);
localparam integer ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer AXI_ADDR_BITS = ADDR_LSB + ADDR_MSB;

/////////////////////////////////////
//          REGISTERS             //
///////////////////////////////////
// Internal AXI registers
logic [AXI_ADDR_BITS-1:0] axi_awaddr;
logic axi_awready;
logic [AXI_ADDR_BITS-1:0] axi_araddr;
logic axi_arready;
logic [1:0] axi_bresp;
logic axi_bvalid;
logic axi_wready;
logic [AXIL_DATA_BITS-1:0] axi_rdata;
logic [1:0] axi_rresp;
logic axi_rvalid;
logic aw_en;

// Registers for holding the values read from/to be written to the AXI Lite interface
// These are synchronous but the outputs are combinatorial
logic [N_REGS-1:0][AXIL_DATA_BITS-1:0] ctrl_reg;
logic ctrl_reg_rden;
logic ctrl_reg_wren;

/////////////////////////////////////
//         REGISTER MAP           //
///////////////////////////////////

// 0 (RO)  : First vaddr for scatter operation
localparam integer BENCH_VADDR_1_REG = 0;

// 1 (RO)   : Second vaddr for scatter operation
localparam integer BENCH_VADDR_2_REG = 1;

// 2 (RO)   : Third vaddr for scatter operation
localparam integer BENCH_VADDR_3_REG = 2;

// 3 (RO)   : Fourth vaddr for scatter operation
localparam integer BENCH_VADDR_4_REG = 3;

// 4 (RO)   : Indicates that all the vaddr for scatter operation are valid
localparam integer BENCH_VADDR_VALID_REG = 4;


/////////////////////////////////////
//         WRITE PROCESS          //
///////////////////////////////////
// Data coming in from host to the vFPGA vie PCIe and XDMA
assign ctrl_reg_wren = axi_wready && axi_ctrl.wvalid && axi_awready && axi_ctrl.awvalid;

always_ff @(posedge aclk) begin
  if (aresetn == 1'b0) begin
    ctrl_reg <= 0;
  end
  else begin
    // Control -> Forcing this to zero is a problem as we want to keep the vaddr values for scatter operation

    if(ctrl_reg_wren) begin
      case (axi_awaddr[ADDR_LSB+:ADDR_MSB])
        BENCH_VADDR_1_REG:     // First vaddr for scatter operation 
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              ctrl_reg[BENCH_VADDR_1_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        BENCH_VADDR_2_REG:    // Second vaddr for scatter operation 
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              ctrl_reg[BENCH_VADDR_2_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        BENCH_VADDR_3_REG:      // Third vaddr for scatter operation 
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              ctrl_reg[BENCH_VADDR_3_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        BENCH_VADDR_4_REG:      // Fourth vaddr for scatter operation 
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              ctrl_reg[BENCH_VADDR_4_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        BENCH_VADDR_VALID_REG:   // Signal that the transmission of all vaddr is completed 
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              ctrl_reg[BENCH_VADDR_VALID_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        default: ;
      endcase
    end
  end
end    


/////////////////////////////////////
//         READ PROCESS           //
///////////////////////////////////
assign ctrl_reg_rden = axi_arready & axi_ctrl.arvalid & ~axi_rvalid;

// There is no actual read process as all registers are write-only, but we need to assign the values to avoid logic trimming 
always_ff @(posedge aclk) begin 
  if(aresetn == 1'b0) begin 
    axi_rdata <= 0;
  end else begin 
    if(ctrl_reg_rden) begin 
      axi_rdata <= 0;
    end 
  end 
end 

/////////////////////////////////////
//       OUTPUT ASSIGNMENT        //
///////////////////////////////////
always_comb begin
  bench_vaddr_1         = ctrl_reg[BENCH_VADDR_1_REG][VADDR_BITS-1:0];
  bench_vaddr_2         = ctrl_reg[BENCH_VADDR_2_REG][VADDR_BITS-1:0];
  bench_vaddr_3         = ctrl_reg[BENCH_VADDR_3_REG][VADDR_BITS-1:0];
  bench_vaddr_4         = ctrl_reg[BENCH_VADDR_4_REG][VADDR_BITS-1:0];
  bench_vaddr_valid     = ctrl_reg[BENCH_VADDR_VALID_REG][0:0];
end

/////////////////////////////////////
//     STANDARD AXI CONTROL       //
///////////////////////////////////
// NOT TO BE EDITED

// I/O
assign axi_ctrl.awready = axi_awready;
assign axi_ctrl.arready = axi_arready;
assign axi_ctrl.bresp = axi_bresp;
assign axi_ctrl.bvalid = axi_bvalid;
assign axi_ctrl.wready = axi_wready;
assign axi_ctrl.rdata = axi_rdata;
assign axi_ctrl.rresp = axi_rresp;
assign axi_ctrl.rvalid = axi_rvalid;

// awready and awaddr
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_awready <= 1'b0;
      axi_awaddr <= 0;
      aw_en <= 1'b1;
    end 
  else
    begin    
      if (~axi_awready && axi_ctrl.awvalid && axi_ctrl.wvalid && aw_en)
        begin
          axi_awready <= 1'b1;
          aw_en <= 1'b0;
          axi_awaddr <= axi_ctrl.awaddr;
        end
      else if (axi_ctrl.bready && axi_bvalid)
        begin
          aw_en <= 1'b1;
          axi_awready <= 1'b0;
        end
      else           
        begin
          axi_awready <= 1'b0;
        end
    end 
end  

// arready and araddr
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_arready <= 1'b0;
      axi_araddr  <= 0;
    end 
  else
    begin    
      if (~axi_arready && axi_ctrl.arvalid)
        begin
          axi_arready <= 1'b1;
          axi_araddr  <= axi_ctrl.araddr;
        end
      else
        begin
          axi_arready <= 1'b0;
        end
    end 
end    

// bvalid and bresp
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_bvalid  <= 0;
      axi_bresp   <= 2'b0;
    end 
  else
    begin    
      if (axi_awready && axi_ctrl.awvalid && ~axi_bvalid && axi_wready && axi_ctrl.wvalid)
        begin
          axi_bvalid <= 1'b1;
          axi_bresp  <= 2'b0;
        end                   
      else
        begin
          if (axi_ctrl.bready && axi_bvalid) 
            begin
              axi_bvalid <= 1'b0; 
            end  
        end
    end
end

// wready
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_wready <= 1'b0;
    end 
  else
    begin    
      if (~axi_wready && axi_ctrl.wvalid && axi_ctrl.awvalid && aw_en )
        begin
          axi_wready <= 1'b1;
        end
      else
        begin
          axi_wready <= 1'b0;
        end
    end 
end  

// rvalid and rresp
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_rvalid <= 0;
      axi_rresp  <= 0;
    end 
  else
    begin    
      if (axi_arready && axi_ctrl.arvalid && ~axi_rvalid)
        begin
          axi_rvalid <= 1'b1;
          axi_rresp  <= 2'b0;
        end   
      else if (axi_rvalid && axi_ctrl.rready)
        begin
          axi_rvalid <= 1'b0;
        end                
    end
end    

endmodule