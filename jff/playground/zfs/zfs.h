#ifndef ZFS_H
#define ZFS_H
#include <cstdint>

// using nvlist = std::uint64_t;

typedef struct nvlist
{
    std::uint64_t id;
    std::uint64_t guid;
    std::string path;
    std::string devid;
    std::uint64_t metaslab_array;
    std::uint64_t metaslab_shift;
    std::uint64_t ashift;
    std::uint64_t asize;
    nvlist_t children[];
} nvlist_t;

// 8K of blank space.
typedef struct blank_space
{
    std::uint8_t empty[8 * 1024];
} blank_space_t;

typedef struct boot_header
{
    std::uint8_t boot[8 * 1024];
} boot_header_t;

typedef struct vdev_label
{
    std::uint64_t version;
    std::string name;
    std::uint64_t state;
    std::uint64_t txg; // transaction group number.
    std::uint64_t pool_guid;
    std::uint64_t top_guid;
    std::uint64_t guid;
    nvlist vdev_tree[];
} vdev_label_t;

// phy vdev and logic vdev

const std::uint8_t POOL_STATE_ACTIVATE = 0x0;
const std::uint8_t POOL_STATE_EXPORTED = 0x1;
const std::uint8_t POOL_STATE_DESTROYED = 0x2;

// UberBlock
const std::uint64_t UB_BIG_ENDIAN = 0x00bab10c;
const std::uint64_t UB_LITTLE_ENDIAN = 0x0cb1ba00;

typedef struct uber_block
{
    std::uint64_t ub_magic;
    std::uint64_t ub_version;
    std::uint64_t ub_txg;
    std::uint64_t ub_guid_sum;
    std::uint64_t ub_timestamp;
    blk_ptr_t ub_rootbp;
} uber_block_t;

typedef struct dva
{
    std::uint64_t data[2]; // todo
} dva_t;

typedef struct checksum_256
{
    std::uint64_t checksum[4];
} checksum_256_t;

typedef struct blk_ptr
{
    dva_t dva[3];
    std::uint64_t info; // endian, type, checksum etc...
    std::uint64_t padding[3];
    std::uint64_t birth_txg;
    std::uint64_t fill_content;
    checksum_256_t checksum;
} blk_ptr_t;

const std::uint8_t DMU_OT_NONE = 0;
const std::uint8_t DMU_OT_OBJECT_DIRECTORY = 1;
const std::uint8_t DMU_OT_OBJECT_ARRAY = 2;
const std::uint8_t DMU_OT_PACKED_NVLIST = 3;
const std::uint8_t DMU_OT_NVLIST_SIZE = 4;
const std::uint8_t DMU_OT_BPLIST = 5;
const std::uint8_t DMU_OT_BPLIST_HDR = 6;
const std::uint8_t DMU_OT_SPACE_MAP_HEADER = 7;
const std::uint8_t DMU_OT_SPACE_MAP = 8;
const std::uint8_t DMU_OT_INTENT_LOG = 9;
const std::uint8_t DMU_OT_DNODE = 10;
const std::uint8_t DMU_OT_OBJECT = 11;
const std::uint8_t DMU_OT_DSL_DATASET = 12;
const std::uint8_t DMU_OT_DSL_DATASET_CHILD_MAP = 13;
const std::uint8_t DMU_OT_OBJECT_SNAP_MAP = 14;
const std::uint8_t DMU_OT_DSL_PROPS = 15;
const std::uint8_t DMU_OT_DSL_OBJSET = 16;
const std::uint8_t DMU_OT_ZNODE = 17;
const std::uint8_t DMU_OT_ACL = 18;
const std::uint8_t DMU_OT_PLAIN_FILE_CONTENT = 19;
const std::uint8_t DMU_OT_DIRECTORY_CONTENT = 20;
const std::uint8_t DMU_OT_MASTER_NODE = 21;
const std::uint8_t DMU_OT_DELETE_QUEUE = 22;
const std::uint8_t DMU_OT_ZVOL = 23;
const std::uint8_t DMU_OT_ZVOL_PROP = 24;

typedef struct dnode_phys
{
    std::uint8_t dn_type;
    std::uint8_t dn_indblk_shift;
    std::uint8_t dn_nlevels;
    std::uint8_t dn_nblkptr;
    std::uint8_t dn_bonustype;
    std::uint8_t dn_checksum;
    std::uint8_t dn_compress;
    std::uint8_t dn_pad;


} dnode_phys_t;

#endif // ZFS_H