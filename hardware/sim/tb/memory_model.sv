`timescale 1ns/1fs
`ifdef XILINX_VIP
import axi_vip_pkg::*;
import fi_mem_vip_pkg::*;
`endif
module fi_memory_model #(parameter integer DW=256,MEMORY_BYTES=2097152)(
    input wire clk,rst,
    input wire [3:0] awid,arid,output wire [3:0] bid,rid,
    input wire awvalid,output wire awready,input wire wvalid,output wire wready,input wire wlast,
    output wire bvalid,input wire bready,input wire arvalid,output wire arready,
    output wire rvalid,input wire rready,output wire rlast,
    input wire [63:0] awaddr,araddr,input wire [7:0] awlen,arlen,
    input wire [2:0] awsize,arsize,input wire [1:0] awburst,arburst,output wire [1:0] bresp,rresp,
    input wire [DW-1:0] wdata,output wire [DW-1:0] rdata,input wire [DW/8-1:0] wstrb
);
    localparam integer BYTES=DW/8;
    integer read_bursts=0,write_bursts=0,read_outstanding=0,write_outstanding=0;
    integer read_peak=0,write_peak=0;
    reg initialized=0;
    always @(posedge clk)if(!rst)begin
        if(arvalid && arready)begin read_bursts=read_bursts+1;read_outstanding=read_outstanding+1;end
        if(rvalid && rready && rlast)read_outstanding=read_outstanding-1;
        if(awvalid && awready)begin write_bursts=write_bursts+1;write_outstanding=write_outstanding+1;end
        if(bvalid && bready)write_outstanding=write_outstanding-1;
        if(read_outstanding>read_peak)read_peak=read_outstanding;
        if(write_outstanding>write_peak)write_peak=write_outstanding;
    end
`ifdef XILINX_VIP
    fi_mem_vip vip (
        .aclk(clk),.aresetn(!rst),
        .s_axi_awid(awid),.s_axi_awaddr(awaddr),.s_axi_awlen(awlen),.s_axi_awsize(awsize),.s_axi_awburst(awburst),
        .s_axi_awvalid(awvalid),.s_axi_awready(awready),.s_axi_wdata(wdata),.s_axi_wstrb(wstrb),
        .s_axi_wlast(wlast),.s_axi_wvalid(wvalid),.s_axi_wready(wready),
        .s_axi_bid(bid),.s_axi_bresp(bresp),.s_axi_bvalid(bvalid),.s_axi_bready(bready),
        .s_axi_arid(arid),.s_axi_araddr(araddr),.s_axi_arlen(arlen),.s_axi_arsize(arsize),.s_axi_arburst(arburst),
        .s_axi_arvalid(arvalid),.s_axi_arready(arready),.s_axi_rid(rid),.s_axi_rdata(rdata),
        .s_axi_rresp(rresp),.s_axi_rlast(rlast),.s_axi_rvalid(rvalid),.s_axi_rready(rready));
    fi_mem_vip_slv_mem_t agent;
    initial begin
        axi_ready_gen ready_gen;
        agent=new("dram",vip.inst.IF);agent.start_slave();
        ready_gen=new("random_wready");ready_gen.set_ready_policy(XIL_AXI_READY_GEN_RANDOM);
        ready_gen.set_low_time_range(0,7);ready_gen.set_high_time_range(1,8);
        agent.wr_driver.set_wready_gen(ready_gen);
        agent.mem_model.set_inter_beat_gap_delay_policy(XIL_AXI_MEMORY_DELAY_RANDOM);
        agent.mem_model.set_inter_beat_gap_range(0,6);
        initialized=1;
    end
    task automatic put(input reg [63:0] addr,input reg [31:0] value);
        agent.mem_model.backdoor_memory_write_4byte(addr,value);
    endtask
    function automatic [31:0] get(input reg [63:0] addr);
        return agent.mem_model.backdoor_memory_read_4byte(addr);
    endfunction
`else
    reg [7:0] mem[0:MEMORY_BYTES-1];
    reg [31:0] random_state=32'habc12345;
    reg read_busy=0,write_busy=0,rv=0,bv=0;
    reg [63:0] ra=0,wa=0;
    reg [3:0] ri=0,wi=0;
    integer rn=0,rb=0,wn=0,wb=0;
    reg [DW-1:0] rd=0;
    reg last_read=0;
    initial initialized=1;
    assign arready=!rst && !read_busy && !rv && random_state[0];
    assign awready=!rst && !write_busy && !bv && random_state[1];
    assign wready=!rst && write_busy && random_state[2];
    assign rvalid=rv;assign rdata=rd;assign rid=ri;assign rresp=0;assign rlast=last_read;
    assign bvalid=bv;assign bid=wi;assign bresp=0;
    task automatic put(input reg [63:0] addr,input reg [31:0] value);
        for(integer n=0;n<4;n=n+1)mem[addr+n]=value[n*8+:8];
    endtask
    function automatic [31:0] get(input reg [63:0] addr);
        return {mem[addr+3],mem[addr+2],mem[addr+1],mem[addr]};
    endfunction
    always @(posedge clk)begin
        random_state<=((random_state<<1) ^ (random_state[31] ? 32'h04c11db7 : 0));
        if(rst)begin read_busy<=0;write_busy<=0;rv<=0;bv<=0;end
        else begin
            if(arvalid && arready)begin
                if(arsize!=$clog2(BYTES) || arburst!=1 || araddr%BYTES!=0 ||
                   araddr[11:0]+(32'(arlen)+1)*BYTES>4096)$fatal(1,"Illegal AXI read burst");
                ra<=araddr;ri<=arid;rn<=32'(arlen)+1;rb<=0;read_busy<=1;
            end
            if(rv && rready)begin rv<=0;rb<=rb+1;if(last_read)read_busy<=0;end
            if(read_busy && !rv && random_state[3])begin
                for(integer n=0;n<BYTES;n=n+1)rd[n*8+:8]<=mem[ra+64'(rb*BYTES+n)];
                last_read<=rb==rn-1;rv<=1;
            end
            if(awvalid && awready)begin
                if(awsize!=$clog2(BYTES) || awburst!=1 || awaddr%BYTES!=0 ||
                   awaddr[11:0]+(32'(awlen)+1)*BYTES>4096)$fatal(1,"Illegal AXI write burst");
                wa<=awaddr;wi<=awid;wn<=32'(awlen)+1;wb<=0;write_busy<=1;
            end
            if(wvalid && wready)begin
                if(wlast!=(wb==wn-1))$fatal(1,"WLAST mismatch");
                for(integer n=0;n<BYTES;n=n+1)if(wstrb[n])mem[wa+64'(wb*BYTES+n)]<=wdata[n*8+:8];
                if(wlast)begin write_busy<=0;bv<=1;end
                else wb<=wb+1;
            end
            if(bv && bready)bv<=0;
        end
    end
`endif
endmodule
