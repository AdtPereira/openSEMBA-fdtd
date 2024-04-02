#include <gtest/gtest.h>

extern "C" int test_idchildtable_fhash();
extern "C" int test_idchildtable();

extern "C" int test_cells();
extern "C" int test_mesh_add_get();
extern "C" int test_mesh_node_to_pixel();
extern "C" int test_mesh_polyline_to_linel();

extern "C" int test_parser_ctor();

extern "C" int test_read_planewave();
extern "C" int test_read_holland1981();
extern "C" int test_read_currentinjection();
extern "C" int test_read_shieldedpair();
extern "C" int test_read_mtln();
extern "C" int test_read_sphere();

TEST(smbjson, idchildtable_fhash)     {EXPECT_EQ(0, test_idchildtable_fhash()); }
TEST(smbjson, idchildtable_add_get)   {EXPECT_EQ(0, test_idchildtable()); }

TEST(smbjson, mesh_cells)             { EXPECT_EQ(0, test_cells()); }
TEST(smbjson, mesh_add_get)           { EXPECT_EQ(0, test_mesh_add_get()); }
TEST(smbjson, mesh_node_to_pixel)     { EXPECT_EQ(0, test_mesh_node_to_pixel()); }
TEST(smbjson, mesh_polyline_to_linel) { EXPECT_EQ(0, test_mesh_polyline_to_linel()); }

TEST(smbjson, parser_ctor)            { EXPECT_EQ(0, test_parser_ctor()); }
TEST(smbjson, read_planewave)         { EXPECT_EQ(0, test_read_planewave()); }
TEST(smbjson, read_holland1981)       { EXPECT_EQ(0, test_read_holland1981()); }
TEST(smbjson, read_currentinjection)  { EXPECT_EQ(0, test_read_currentinjection()); }
TEST(smbjson, read_shieldedpair)      { EXPECT_EQ(0, test_read_shieldedpair()); }
TEST(smbjson, read_mtln)              { EXPECT_EQ(0, test_read_mtln()); }
TEST(smbjson, read_sphere)            { EXPECT_EQ(0, test_read_sphere()); }
