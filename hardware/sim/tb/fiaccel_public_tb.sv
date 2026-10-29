`timescale 1ns/1fs
`default_nettype none
`ifndef AXI_DW
`define AXI_DW 256
`endif
`ifdef XILINX_VIP
import axi_vip_pkg::*;
import fi_mem_vip_pkg::*;
import fi_ctrl_vip_pkg::*;
`endif
module fiaccel_public_tb;
    `include "fixture.vh"
    localparam integer DW=`AXI_DW,BYTES=DW/8;
    reg clk=0;always #1.666667 clk=~clk;
    reg rst=1;
    wire ap_done,ap_idle,ap_ready,error;
    wire [31:0] fmr_writes,fmr_skips;
    wire [31:0] cache_hits,cache_misses;
    wire [4-1:0] awid[0:2];
    wire [4-1:0] arid[0:2];
    wire [4-1:0] bid[0:2];
    wire [4-1:0] rid[0:2];
    wire awvalid[0:2];
    wire awready[0:2];
    wire wvalid[0:2];
    wire wready[0:2];
    wire wlast[0:2];
    wire bvalid[0:2];
    wire bready[0:2];
    wire arvalid[0:2];
    wire arready[0:2];
    wire rvalid[0:2];
    wire rready[0:2];
    wire rlast[0:2];
    wire [64-1:0] awaddr[0:2];
    wire [64-1:0] araddr[0:2];
    wire [8-1:0] awlen[0:2];
    wire [8-1:0] arlen[0:2];
    wire [3-1:0] awsize[0:2];
    wire [3-1:0] arsize[0:2];
    wire [2-1:0] awburst[0:2];
    wire [2-1:0] arburst[0:2];
    wire [2-1:0] bresp[0:2];
    wire [2-1:0] rresp[0:2];
    wire [DW-1:0] wdata[0:2];
    wire [DW-1:0] rdata[0:2];
    wire [BYTES-1:0] wstrb[0:2];
    wire c_awvalid,c_awready,c_wvalid,c_wready,c_bvalid,c_bready;
    wire c_arvalid,c_arready,c_rvalid,c_rready;
    wire [11:0] c_awaddr,c_araddr;
    wire [31:0] c_wdata,c_rdata;
    wire [3:0] c_wstrb;
    wire [1:0] c_bresp,c_rresp;
    wire interrupt;
`ifdef FIACCEL_SIM_DEBUG
    wire [31:0] debug_encoder_x;
    wire [31:0] debug_encoder_y;
    wire [47:0] debug_flow;
    wire [0:0] debug_flow_valid;
    wire [0:0] debug_flow_ready;
    wire [31:0] debug_cycles;
    wire [7:0] debug_state;
`endif
    FIACCEL #(.W(W),.H(H),.D0(D0),.D1(D1),.D2(D2),.D3(D3),.D4(D4),.D5(D5),.D6(D6),
        .C_M00_AXI_DATA_WIDTH(DW),.C_M01_AXI_DATA_WIDTH(DW),.C_M02_AXI_DATA_WIDTH(DW)) dut (
`ifdef FIACCEL_SIM_DEBUG
        .debug_encoder_x(debug_encoder_x),
        .debug_encoder_y(debug_encoder_y),
        .debug_flow(debug_flow),
        .debug_flow_valid(debug_flow_valid),
        .debug_flow_ready(debug_flow_ready),
        .debug_cycles(debug_cycles),
        .debug_state(debug_state),
`endif
        .ap_clk(clk),.ap_rst_n(!rst),.interrupt(interrupt),
        .s_axi_control_awvalid(c_awvalid),.s_axi_control_awready(c_awready),.s_axi_control_awaddr(c_awaddr),
        .s_axi_control_wvalid(c_wvalid),.s_axi_control_wready(c_wready),.s_axi_control_wdata(c_wdata),.s_axi_control_wstrb(c_wstrb),
        .s_axi_control_bvalid(c_bvalid),.s_axi_control_bready(c_bready),.s_axi_control_bresp(c_bresp),
        .s_axi_control_arvalid(c_arvalid),.s_axi_control_arready(c_arready),.s_axi_control_araddr(c_araddr),
        .s_axi_control_rvalid(c_rvalid),.s_axi_control_rready(c_rready),.s_axi_control_rdata(c_rdata),.s_axi_control_rresp(c_rresp),
        .error(error),.fmr_writes(fmr_writes),.fmr_skips(fmr_skips),.cache_hits(cache_hits),.cache_misses(cache_misses),
        .m00_axi_awid(awid[0]),.m00_axi_arid(arid[0]),.m00_axi_bid(bid[0]),.m00_axi_rid(rid[0]),.m00_axi_awvalid(awvalid[0]),.m00_axi_awready(awready[0]),.m00_axi_wvalid(wvalid[0]),.m00_axi_wready(wready[0]),.m00_axi_wlast(wlast[0]),.m00_axi_bvalid(bvalid[0]),.m00_axi_bready(bready[0]),.m00_axi_arvalid(arvalid[0]),.m00_axi_arready(arready[0]),.m00_axi_rvalid(rvalid[0]),.m00_axi_rready(rready[0]),.m00_axi_rlast(rlast[0]),.m00_axi_awaddr(awaddr[0]),.m00_axi_araddr(araddr[0]),.m00_axi_awlen(awlen[0]),.m00_axi_arlen(arlen[0]),.m00_axi_awsize(awsize[0]),.m00_axi_arsize(arsize[0]),.m00_axi_awburst(awburst[0]),.m00_axi_arburst(arburst[0]),.m00_axi_bresp(bresp[0]),.m00_axi_rresp(rresp[0]),.m00_axi_wdata(wdata[0]),.m00_axi_rdata(rdata[0]),.m00_axi_wstrb(wstrb[0]),
        .m01_axi_awid(awid[1]),.m01_axi_arid(arid[1]),.m01_axi_bid(bid[1]),.m01_axi_rid(rid[1]),.m01_axi_awvalid(awvalid[1]),.m01_axi_awready(awready[1]),.m01_axi_wvalid(wvalid[1]),.m01_axi_wready(wready[1]),.m01_axi_wlast(wlast[1]),.m01_axi_bvalid(bvalid[1]),.m01_axi_bready(bready[1]),.m01_axi_arvalid(arvalid[1]),.m01_axi_arready(arready[1]),.m01_axi_rvalid(rvalid[1]),.m01_axi_rready(rready[1]),.m01_axi_rlast(rlast[1]),.m01_axi_awaddr(awaddr[1]),.m01_axi_araddr(araddr[1]),.m01_axi_awlen(awlen[1]),.m01_axi_arlen(arlen[1]),.m01_axi_awsize(awsize[1]),.m01_axi_arsize(arsize[1]),.m01_axi_awburst(awburst[1]),.m01_axi_arburst(arburst[1]),.m01_axi_bresp(bresp[1]),.m01_axi_rresp(rresp[1]),.m01_axi_wdata(wdata[1]),.m01_axi_rdata(rdata[1]),.m01_axi_wstrb(wstrb[1]),
        .m02_axi_awid(awid[2]),.m02_axi_arid(arid[2]),.m02_axi_bid(bid[2]),.m02_axi_rid(rid[2]),.m02_axi_awvalid(awvalid[2]),.m02_axi_awready(awready[2]),.m02_axi_wvalid(wvalid[2]),.m02_axi_wready(wready[2]),.m02_axi_wlast(wlast[2]),.m02_axi_bvalid(bvalid[2]),.m02_axi_bready(bready[2]),.m02_axi_arvalid(arvalid[2]),.m02_axi_arready(arready[2]),.m02_axi_rvalid(rvalid[2]),.m02_axi_rready(rready[2]),.m02_axi_rlast(rlast[2]),.m02_axi_awaddr(awaddr[2]),.m02_axi_araddr(araddr[2]),.m02_axi_awlen(awlen[2]),.m02_axi_arlen(arlen[2]),.m02_axi_awsize(awsize[2]),.m02_axi_arsize(arsize[2]),.m02_axi_awburst(awburst[2]),.m02_axi_arburst(arburst[2]),.m02_axi_bresp(bresp[2]),.m02_axi_rresp(rresp[2]),.m02_axi_wdata(wdata[2]),.m02_axi_rdata(rdata[2]),.m02_axi_wstrb(wstrb[2]));


`ifdef XILINX_VIP
    fi_ctrl_vip control_vip (
        .aclk(clk),.aresetn(!rst),
        .m_axi_awvalid(c_awvalid),.m_axi_awready(c_awready),.m_axi_awaddr(c_awaddr),
        .m_axi_wvalid(c_wvalid),.m_axi_wready(c_wready),.m_axi_wdata(c_wdata),.m_axi_wstrb(c_wstrb),
        .m_axi_bvalid(c_bvalid),.m_axi_bready(c_bready),.m_axi_bresp(c_bresp),
        .m_axi_arvalid(c_arvalid),.m_axi_arready(c_arready),.m_axi_araddr(c_araddr),
        .m_axi_rvalid(c_rvalid),.m_axi_rready(c_rready),.m_axi_rdata(c_rdata),.m_axi_rresp(c_rresp));
    fi_ctrl_vip_mst_t ctrl;
    task automatic control_write(input reg [31:0] addr_in,input reg [31:0] value);
        axi_transaction tx,rsp;
        tx=ctrl.wr_driver.create_transaction("control_write");
        tx.set_driver_return_item_policy(XIL_AXI_PAYLOAD_RETURN);
        assert(tx.randomize() with {addr==addr_in;});
        tx.set_data_beat(0,value);ctrl.wr_driver.send(tx);ctrl.wr_driver.wait_rsp(rsp);
    endtask
    task automatic control_read(input reg [31:0] addr_in,output reg [31:0] value);
        axi_transaction tx,rsp;
        tx=ctrl.rd_driver.create_transaction("control_read");tx.set_addr(addr_in);
        tx.set_driver_return_item_policy(XIL_AXI_PAYLOAD_RETURN);
        ctrl.rd_driver.send(tx);ctrl.rd_driver.wait_rsp(rsp);value=rsp.get_data_beat(0);
    endtask
`else
    reg ca_valid=0,cw_valid=0,cr_valid=0,cb_ready=0,cr_ready=0;
    reg [11:0] ca_addr=0,cr_addr=0;
    reg [31:0] cw_data=0;
    assign c_awvalid=ca_valid;assign c_awaddr=ca_addr;
    assign c_wvalid=cw_valid;assign c_wdata=cw_data;assign c_wstrb=4'hf;
    assign c_bready=cb_ready;assign c_arvalid=cr_valid;assign c_araddr=cr_addr;assign c_rready=cr_ready;
    task automatic control_write(input reg [31:0] addr_in,input reg [31:0] value);
        @(negedge clk);ca_addr=addr_in[11:0];ca_valid=1;
        do @(posedge clk);while(!c_awready);
        @(negedge clk);ca_valid=0;cw_data=value;cw_valid=1;
        do @(posedge clk);while(!c_wready);
        @(negedge clk);cw_valid=0;cb_ready=1;
        do @(posedge clk);while(!c_bvalid);
        if(c_bresp!=0)$fatal(1,"AXI-Lite write failed");
        @(negedge clk);cb_ready=0;
    endtask
    task automatic control_read(input reg [31:0] addr_in,output reg [31:0] value);
        @(negedge clk);cr_addr=addr_in[11:0];cr_valid=1;
        do @(posedge clk);while(!c_arready);
        @(negedge clk);cr_valid=0;cr_ready=1;
        do @(posedge clk);while(!c_rvalid);
        value=c_rdata;if(c_rresp!=0)$fatal(1,"AXI-Lite read failed");
        @(negedge clk);cr_ready=0;
    endtask
`endif
    fi_memory_model #(.DW(DW),.MEMORY_BYTES(MEMORY_BYTES)) memory0 (.clk(clk),.rst(rst),.awid(awid[0]),.arid(arid[0]),.bid(bid[0]),.rid(rid[0]),.awvalid(awvalid[0]),.awready(awready[0]),.wvalid(wvalid[0]),.wready(wready[0]),.wlast(wlast[0]),.bvalid(bvalid[0]),.bready(bready[0]),.arvalid(arvalid[0]),.arready(arready[0]),.rvalid(rvalid[0]),.rready(rready[0]),.rlast(rlast[0]),.awaddr(awaddr[0]),.araddr(araddr[0]),.awlen(awlen[0]),.arlen(arlen[0]),.awsize(awsize[0]),.arsize(arsize[0]),.awburst(awburst[0]),.arburst(arburst[0]),.bresp(bresp[0]),.rresp(rresp[0]),.wdata(wdata[0]),.rdata(rdata[0]),.wstrb(wstrb[0]));
    fi_memory_model #(.DW(DW),.MEMORY_BYTES(MEMORY_BYTES)) memory1 (.clk(clk),.rst(rst),.awid(awid[1]),.arid(arid[1]),.bid(bid[1]),.rid(rid[1]),.awvalid(awvalid[1]),.awready(awready[1]),.wvalid(wvalid[1]),.wready(wready[1]),.wlast(wlast[1]),.bvalid(bvalid[1]),.bready(bready[1]),.arvalid(arvalid[1]),.arready(arready[1]),.rvalid(rvalid[1]),.rready(rready[1]),.rlast(rlast[1]),.awaddr(awaddr[1]),.araddr(araddr[1]),.awlen(awlen[1]),.arlen(arlen[1]),.awsize(awsize[1]),.arsize(arsize[1]),.awburst(awburst[1]),.arburst(arburst[1]),.bresp(bresp[1]),.rresp(rresp[1]),.wdata(wdata[1]),.rdata(rdata[1]),.wstrb(wstrb[1]));
    fi_memory_model #(.DW(DW),.MEMORY_BYTES(MEMORY_BYTES)) memory2 (.clk(clk),.rst(rst),.awid(awid[2]),.arid(arid[2]),.bid(bid[2]),.rid(rid[2]),.awvalid(awvalid[2]),.awready(awready[2]),.wvalid(wvalid[2]),.wready(wready[2]),.wlast(wlast[2]),.bvalid(bvalid[2]),.bready(bready[2]),.arvalid(arvalid[2]),.arready(arready[2]),.rvalid(rvalid[2]),.rready(rready[2]),.rlast(rlast[2]),.awaddr(awaddr[2]),.araddr(araddr[2]),.awlen(awlen[2]),.arlen(arlen[2]),.awsize(awsize[2]),.arsize(arsize[2]),.awburst(awburst[2]),.arburst(arburst[2]),.bresp(bresp[2]),.rresp(rresp[2]),.wdata(wdata[2]),.rdata(rdata[2]),.wstrb(wstrb[2]));
    reg [63:0] initial_words[0:INIT_WORDS-1];
    reg [31:0] expected[0:(FRAMES-1)*W*H-1];
    reg [255:0] cache_expected[0:W*H/64-1];
    integer cycles=0,output_fd;
    integer coefficient_reads=0,coefficient_reads_at_config=0;
    reg coefficients_loaded=0;
    task automatic execute(input reg [63:0] pointer,input reg [31:0] mode);
        reg [31:0] status;
        control_write(32'h10,mode);control_write(32'h24,pointer[31:0]);control_write(32'h28,pointer[63:32]);
        control_write(0,1);
        do begin repeat(64) @(posedge clk);control_read(0,status);end while(!status[1]);
        control_read(32'h30,status);if(status!=0 || error)$fatal(1,"Kernel error status");
        control_read(32'h34,status);if(status!=fmr_writes)$fatal(1,"FMR write register mismatch");
        control_read(32'h38,status);if(status!=fmr_skips)$fatal(1,"FMR skip register mismatch");
        control_read(32'h44,status);
        if(status==0)$fatal(1,"Command cycle readback mismatch");
        $display("COMMAND_CYCLES mode=%08x clocks=%0d",mode,status);
    endtask
    initial begin: run
        reg [31:0] value;
`ifdef XILINX_VIP
        ctrl=new("control",control_vip.inst.IF);ctrl.start_master();
`endif
        wait(memory0.initialized && memory1.initialized && memory2.initialized);
        $readmemh("init.hex",initial_words);$readmemh("expected.hex",expected);
        $readmemh("cache.hex",cache_expected);
        for(integer n=0;n<INIT_WORDS;n=n+1)begin
            memory0.put({32'b0,initial_words[n][63:32]},initial_words[n][31:0]);
            memory1.put({32'b0,initial_words[n][63:32]},initial_words[n][31:0]);
            memory2.put({32'b0,initial_words[n][63:32]},initial_words[n][31:0]);
        end
        output_fd=$fopen("rtl_output.hex","w");
        repeat(32) @(negedge clk);rst=0;repeat(6) @(negedge clk);
        control_write(32'h18,32'(CACHE_BASE));control_write(32'h1c,32'(CACHE_BASE>>32));
        execute(CONFIG_BASE,32'h80000000|32'(RECORDS));
        coefficient_reads_at_config=coefficient_reads;
        coefficients_loaded=1;
        $display("CONFIG_DONE cycles=%0d",cycles);
        for(integer frame=0;frame<FRAMES;frame=frame+1) begin
            control_write(32'h3c,32'(output_base(frame>0 ? frame-1 : 0)));
            control_write(32'h40,32'(output_base(frame>0 ? frame-1 : 0)>>32));
            execute(descriptor(frame),0);
            $display("FRAME_DONE frame=%0d cycles=%0d writes=%0d skips=%0d",frame,cycles,fmr_writes,fmr_skips);
            if(fmr_writes!=gold_writes(frame) || fmr_skips!=gold_skips(frame))
                $fatal(1,"Selective transfer counts frame=%0d got %0d/%0d expected %0d/%0d",frame,
                    fmr_writes,fmr_skips,gold_writes(frame),gold_skips(frame));
            if(frame>0) for(integer p=0;p<W*H;p=p+1) begin
                value=memory2.get(output_base(frame-1)+64'(p)*4);
                if(value!==expected[(frame-1)*W*H+p])
                    $fatal(1,"Image frame=%0d pixel=%0d got=%h expected=%h",frame,p,value,expected[(frame-1)*W*H+p]);
                $fdisplay(output_fd,"%08x",value);
            end
        end
        for(integer p=0;p<W*H/64;p=p+1) for(integer n=0;n<8;n=n+1) begin
            value=memory1.get(CACHE_BASE+64'(p)*32+64'(n)*4);
            if(value!==cache_expected[p][n*32+:32]) $fatal(1,"Persistent FMR representative mismatch");
        end
        if(coefficient_reads!=coefficient_reads_at_config || coefficient_reads==0)
            $fatal(1,"Coefficients must remain resident across frame commands");
        $display("WEIGHT_REUSE config_line_reads=%0d frame_coefficient_reads=%0d frames=%0d",
            coefficient_reads_at_config,coefficient_reads-coefficient_reads_at_config,FRAMES);
        $fclose(output_fd);
        if(cache_hits+cache_misses!=8*(FRAMES-1)*W*H)$fatal(1,"Cache request accounting");
        if(cache_hits==0 || cache_misses==0)$fatal(1,"Cache hit/miss paths not exercised");
        $display("AXI_PORTS read_peaks=%0d,%0d,%0d write_peaks=%0d,%0d,%0d cache_hits=%0d cache_misses=%0d",memory0.read_peak,memory1.read_peak,memory2.read_peak,memory0.write_peak,memory1.write_peak,memory2.write_peak,cache_hits,cache_misses);
        if(memory2.write_outstanding!=0)$fatal(1,"Kernel completed with pending output writes");
        $display("AXI_BURSTS reads=%0d,%0d,%0d writes=%0d,%0d,%0d",memory0.read_bursts,memory1.read_bursts,memory2.read_bursts,memory0.write_bursts,memory1.write_bursts,memory2.write_bursts);
        $display("FIACCEL_PUBLIC_PASS frames=%0d pixels=%0d writes=%0d skips=%0d cycles=%0d",FRAMES,(FRAMES-1)*W*H,fmr_writes,fmr_skips,cycles);
        $finish;
    end
    always @(posedge clk) if(!rst) begin
        cycles<=cycles+1;
        if(arvalid[0] && arready[0] && araddr[0]>=CONFIG_BASE &&
           araddr[0]<CONFIG_BASE+64'(((RECORDS+3)/4)*32)) begin
            if(coefficients_loaded)$fatal(1,"Frame command fetched coefficients from DDR again");
            coefficient_reads=coefficient_reads+1;
        end
        if(error) $fatal(1,"DUT reported an error");
        if(cycles>20000000) $fatal(1,"End-to-end timeout");
    end
endmodule
`default_nettype wire
