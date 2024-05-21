
#define ROWS_PER_XOR_PATTERN 8   // 每一行有8个方格
#define COLS_PER_XOR_PATTERN 1
#define BYTES_PER_STS 16    // 每个方格的大小是128bit
#define BYTES_PER_ROW  BYTES_PER_STS*ROWS_PER_XOR_PATTERN
extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);

struct Smem_tile_row_a{
    inline __device__ Smem_tile_row_a(void *smem, int tidx):smem_(__nvvm_get_smem_pointer(smem)), tidx_(tidx){


        // The number of warps.
        const int WARPS_M = 1;
        const int WARPS_N = 2;
        const int WARPS_K = 1;

        //---------------------------------从global mem读取的数据写入shared mem中----------------------
        //  其实就是（row,col） -> (row, row ^ col)
        // The row written by a thread. See doc/mma_smem_layout.xlsx.
        int smem_write_row = tidx / THREADS_PER_ROW;
        // The XOR pattern.
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN * COLS_PER_XOR_PATTERN;
        // Compute the column and apply the XOR pattern.
        int smem_write_col = (tidx % THREADS_PER_ROW) ^ smem_write_xor;
        // The offset.
        this->smem_write_offset_ = smem_write_row*BYTES_PER_ROW + smem_write_col*BYTES_PER_STS;


        //---------------------------------从shared mem读取的数据写入register中----------------------
        // The row and column read by the thread.
//        int smem_read_row  = (tidx & 0x0f);
//        constexpr int ROWS_PER_PACKING = BYTES_PER_ROW / BYTES_PER_ROW_BEFORE_PACKING;
//        int smem_read_col = ((smem_read_row / ROWS_PER_PACKING) % ROWS_PER_XOR_PATTERN) * Base::COLS_PER_XOR_PATTERN;
//        smem_read_col ^= (tidx & 0x10) / 16;
//
//        // The shared memory offset.
//        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW_BEFORE_PACKING + smem_read_col*BYTES_PER_LDS;
    }


    inline __device__ void store(const uint4 data) {

        // this->compute_store_pointers(smem_ptrs);

        sts(this->smem_ + smem_write_offset_, data);
    }

    // The shared memory pointer.
    const uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    // int smem_read_buffer_;
    // The buffer base offset for write.
    // int smem_write_buffer_;
    const int tidx_;

};