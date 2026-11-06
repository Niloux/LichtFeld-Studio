/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_volume.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "splat_mesh.hpp"
#include <cfloat>
#include <cstring>
#include <format>
#include <fstream>
#include <stb_image_write.h>
#include <string_view>
#include <vector>
#include <zlib.h>

// Writes the small part of OpenSceneGraph's binary format these models use, so that they need no OSG libraries or
// plugins to make. The bytes match what OSG 3.6.5 writes for the same scene graph, field by field: its serializers
// write every field of a class and its base classes in a fixed order, whether or not it holds a default. A reader
// matches them up by position, so the order below is not a choice.
namespace lfs::training {
    namespace {
        using core::collisionError;

        constexpr int kOsgbJpegQuality = 95;
        constexpr uint32_t kOsgVersion = 161; // OSG 3.6.5
        constexpr uint32_t kSceneFile = 1;
        constexpr uint32_t kBinaryBrackets = 4;
        constexpr uint32_t kUnspecifiedVariance = 2, kStaticVariance = 1;
        constexpr uint32_t kCenterUserDefined = 1, kRangePixelSize = 1;
        constexpr uint32_t kBindPerVertex = 4, kTriangles = 4;
        constexpr uint32_t kGlArrayBuffer = 0x8892, kGlElementBuffer = 0x8893, kGlStaticDraw = 0x88e4;
        constexpr uint32_t kGlLighting = 0x0b50, kGlTexture2D = 0x0de1;
        constexpr uint32_t kGlClampToEdge = 0x812f, kGlClamp = 0x2900, kGlLinearMipmapLinear = 0x2703, kGlLinear = 0x2601;
        constexpr uint32_t kGlLequal = 0x0203, kGlLuminance = 0x1909;
        constexpr uint32_t kImageInlineFile = 1;

        // DasViewer's own model layout. The model is in the dataset's local coordinates; this origin only gives it a place.
        constexpr const char* kMetadata = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
                                          "<ModelMetadata version=\"1\">\n"
                                          "\t<SRS>ENU:31.0,121.0</SRS>\n"
                                          "\t<SRSOrigin>0,0,0</SRSOrigin>\n"
                                          "\t<Texture>\n\t\t<ColorSource>Visible</ColorSource>\n\t</Texture>\n"
                                          "</ModelMetadata>\n";

        class OsgbStream {
        public:
            void bytes(const void* data, const size_t size) {
                const auto* begin = static_cast<const char*>(data);
                buffer_.insert(buffer_.end(), begin, begin + size);
            }
            void flag(const bool value) { buffer_.push_back(value ? 1 : 0); }
            void u32(const uint32_t value) { bytes(&value, sizeof(value)); }
            void f32(const float value) { bytes(&value, sizeof(value)); }
            void f64(const double value) { bytes(&value, sizeof(value)); }
            void text(const std::string_view value) {
                u32(static_cast<uint32_t>(value.size()));
                bytes(value.data(), value.size());
            }
            void zeros(const size_t count) { buffer_.resize(buffer_.size() + count, 0); }
            // A bracket holds the size of what it encloses, itself included, so that a reader can skip a class it lacks.
            void open() {
                brackets_.push_back(buffer_.size());
                zeros(sizeof(uint64_t));
            }
            void close() {
                const size_t at = brackets_.back();
                brackets_.pop_back();
                const uint64_t size = buffer_.size() - at;
                std::memcpy(buffer_.data() + at, &size, sizeof(size));
            }
            [[nodiscard]] const std::vector<char>& data() const { return buffer_; }

        private:
            std::vector<char> buffer_;
            std::vector<size_t> brackets_;
        };

        // osg::Object
        void write_object_fields(OsgbStream& out, const uint32_t variance = kUnspecifiedVariance) {
            out.text("");      // name
            out.u32(variance); // data variance
            out.flag(false);   // user data
        }

        // osg::Node, up to its state set
        void write_node_fields(OsgbStream& out) {
            out.zeros(5);         // initial bound, compute bound, update, event and cull callbacks
            out.flag(true);       // culling active
            out.u32(0xffffffffu); // node mask
        }

        // Numbers each object; a reference to one already written is its name and number alone.
        struct ObjectIds {
            uint32_t next = 1;
            uint32_t operator()() { return next++; }
        };

        void write_buffer_object(OsgbStream& out, const std::string_view name, const uint32_t id, const uint32_t target) {
            out.text(name);
            out.open();
            out.u32(id);
            write_object_fields(out);
            out.u32(target);
            out.u32(kGlStaticDraw);
            out.flag(false); // copy data and release
            out.close();
        }

        void write_reference(OsgbStream& out, const std::string_view name, const uint32_t id) {
            out.text(name);
            out.open();
            out.u32(id);
            out.close();
        }

        void write_vertex_array(OsgbStream& out, const std::string_view name, const uint32_t id, uint32_t& buffer_id, ObjectIds& ids, const void* data, const uint32_t count, const size_t element_size) {
            out.text(name);
            out.open();
            out.u32(id);
            write_object_fields(out);
            out.flag(true); // buffer object, one for all arrays of the level
            if (buffer_id == 0) {
                buffer_id = ids();
                write_buffer_object(out, "osg::VertexBufferObject", buffer_id, kGlArrayBuffer);
            } else {
                write_reference(out, "osg::VertexBufferObject", buffer_id);
            }
            out.u32(kBindPerVertex);
            out.zeros(2); // normalize, preserve data type
            out.u32(count);
            out.bytes(data, count * element_size);
            out.close();
        }

        // osg::Texture2D, up to its image: clamped, mipmapped, otherwise as OSG makes them.
        void write_texture_fields(OsgbStream& out) {
            write_object_fields(out);
            out.zeros(2); // update and event callbacks
            for (const uint32_t mode : {kGlClampToEdge, kGlClampToEdge, kGlClamp, kGlLinearMipmapLinear, kGlLinear}) {
                out.flag(true);
                out.u32(mode); // wrap s, t, r; min and mag filter
            }
            out.f32(1.f);       // max anisotropy
            out.flag(true);     // hardware mipmaps
            out.zeros(2);       // unref image, client storage
            out.flag(true);     // resize non power of two
            out.zeros(32);      // border color
            out.zeros(4);       // border width
            out.zeros(4);       // internal format mode
            out.zeros(3);       // internal format, source format, source type
            out.zeros(1);       // shadow comparison
            out.u32(kGlLequal); // shadow compare function
            out.u32(kGlLuminance);
            out.f32(0.f); // shadow ambient
            out.flag(true);
            out.text("RGBA"); // swizzle
            out.f32(0.f);     // min LOD
            out.f32(-1.f);    // max LOD
            out.f32(0.f);     // LOD bias
        }

        [[nodiscard]] std::vector<char> encode_jpeg(const std::vector<uint8_t>& rgb, const int side, const std::string& name) {
            std::vector<char> jpeg;
            const auto append = [](void* context, void* data, const int size) {
                auto* target = static_cast<std::vector<char>*>(context);
                const auto* begin = static_cast<const char*>(data);
                target->insert(target->end(), begin, begin + size);
            };
            if (!stbi_write_jpg_to_func(append, &jpeg, side, side, 3, rgb.data(), kOsgbJpegQuality))
                throw collisionError("mesh.write_failed", "Could not write the mesh texture", ErrorCode::Unavailable, name);
            return jpeg;
        }

        // The faces of one level, one drawable per page, sharing the level's vertices.
        void write_level_geode(OsgbStream& out, ObjectIds& ids, const SplatMesh& all, const TileLevel& level, const std::string& stem) {
            std::vector<std::array<float, 2>> coordinates(level.vertex_count);
            for (size_t v = 0; v < level.vertex_count; ++v) {
                const auto& uv = all.uvs[level.first_vertex + v];
                // A JPEG's first row is the top of the texture, which OSG counts from the bottom.
                coordinates[v] = {uv[0], 1.f - uv[1]};
            }
            out.text("osg::Geode");
            out.open();
            out.u32(ids());
            write_object_fields(out);
            write_node_fields(out);
            out.flag(false); // state set
            out.flag(true);
            out.u32(static_cast<uint32_t>(level.page_count));
            out.open();
            uint32_t vertices_id = 0, normals_id = 0, coordinates_id = 0, buffer_id = 0;
            std::vector<uint32_t> indices;
            for (size_t p = 0; p < level.page_count; ++p) {
                const size_t page = level.first_page + p;
                const auto name = std::format("{}_{}.jpg", stem, p);
                const auto jpeg = encode_jpeg(all.pages[page], all.page_side(page), name);

                out.text("osg::Geometry");
                out.open();
                out.u32(ids());
                write_object_fields(out);
                write_node_fields(out);
                out.flag(true); // state set
                out.text("osg::StateSet");
                out.open();
                out.u32(ids());
                write_object_fields(out);
                // The photos already hold the light of the scene.
                out.flag(true);
                out.u32(1);
                out.open();
                out.u32(kGlLighting);
                out.u32(0); // off
                out.close();
                out.flag(false); // attributes
                out.flag(true);  // texture modes
                out.u32(1);
                out.open();
                out.u32(1);
                out.open();
                out.u32(kGlTexture2D);
                out.u32(1); // on
                out.close();
                out.close();
                out.flag(true); // texture attributes
                out.u32(1);
                out.open();
                out.u32(1);
                out.open();
                out.text("osg::Texture2D");
                out.open();
                out.u32(ids());
                write_texture_fields(out);
                out.flag(true); // image, its JPEG within the file
                out.text("osg::Image");
                out.u32(ids());
                out.text(name);
                out.u32(0); // write hint
                out.u32(kImageInlineFile);
                out.u32(static_cast<uint32_t>(jpeg.size()));
                out.bytes(jpeg.data(), jpeg.size());
                write_object_fields(out, kStaticVariance);
                out.zeros(8); // texture width and height
                out.close();
                out.u32(0); // attribute value
                out.close();
                out.close();
                out.flag(false); // uniforms
                out.zeros(16);   // rendering hint, bin mode, bin number, bin name
                out.flag(true);  // nest render bins
                out.zeros(2);    // update and event callbacks
                out.flag(false); // defines
                out.close();

                // osg::Drawable
                out.zeros(3);         // initial bound, compute bound callback, shape
                out.flag(true);       // supports display list
                out.flag(false);      // use display list
                out.flag(true);       // use vertex buffer objects
                out.u32(0xffffffffu); // node mask
                out.flag(true);       // culling active

                // osg::Geometry
                out.u32(1); // primitive sets
                out.text("osg::DrawElementsUInt");
                out.open();
                out.u32(ids());
                write_object_fields(out);
                out.flag(true);
                write_buffer_object(out, "osg::ElementBufferObject", ids(), kGlElementBuffer);
                out.u32(0); // instances
                out.u32(kTriangles);
                const size_t first = all.page_first_face[page], last = all.page_first_face[page + 1];
                indices.resize((last - first) * 3);
                for (size_t f = first; f < last; ++f)
                    for (size_t c = 0; c < 3; ++c)
                        indices[(f - first) * 3 + c] = static_cast<uint32_t>(size_t(all.faces[f][c]) - level.first_vertex);
                out.u32(static_cast<uint32_t>(indices.size()));
                out.bytes(indices.data(), indices.size() * sizeof(uint32_t));
                out.close();

                const auto count = static_cast<uint32_t>(level.vertex_count);
                out.flag(true);
                if (vertices_id == 0) {
                    vertices_id = ids();
                    write_vertex_array(out, "osg::Vec3Array", vertices_id, buffer_id, ids, all.vertices.data() + level.first_vertex, count, sizeof(float) * 3);
                } else {
                    write_reference(out, "osg::Vec3Array", vertices_id);
                }
                out.flag(true);
                if (normals_id == 0) {
                    normals_id = ids();
                    write_vertex_array(out, "osg::Vec3Array", normals_id, buffer_id, ids, all.normals.data() + level.first_vertex, count, sizeof(float) * 3);
                } else {
                    write_reference(out, "osg::Vec3Array", normals_id);
                }
                out.zeros(3); // color, secondary color and fog coordinate arrays
                out.u32(1);   // texture coordinate arrays
                if (coordinates_id == 0) {
                    coordinates_id = ids();
                    write_vertex_array(out, "osg::Vec2Array", coordinates_id, buffer_id, ids, coordinates.data(), count, sizeof(float) * 2);
                } else {
                    write_reference(out, "osg::Vec2Array", coordinates_id);
                }
                out.u32(0); // vertex attribute arrays
                out.close();
            }
            out.close();
            out.close();
        }

        // Shows this level until the tile spans as many pixels as the level has texels, then the next finer file.
        void write_paged_lod(OsgbStream& out, ObjectIds& ids, const SplatMesh& all, const TileLevel& level, const std::string& stem, const std::string& finer_file) {
            const float full_detail = 2.f * level.radius / level.texel;
            out.text("osg::PagedLOD");
            out.open();
            out.u32(ids());
            write_object_fields(out);
            write_node_fields(out);
            out.flag(false); // state set
            // osg::LOD
            out.u32(kCenterUserDefined);
            out.flag(true);
            for (const float value : {level.center[0], level.center[1], level.center[2], level.radius})
                out.f64(value);
            out.u32(kRangePixelSize);
            out.flag(true);
            out.u32(2);
            out.open();
            for (const float value : {0.f, full_detail, full_detail, FLT_MAX})
                out.f32(value);
            out.close();
            // osg::PagedLOD. No database path: the finer file is found beside this one, wherever the model moves to.
            out.flag(true);
            out.flag(false);
            out.u32(0);      // children that cannot expire
            out.flag(false); // disable external paging
            out.flag(true);
            out.u32(2);
            out.open();
            out.text("");
            out.text(finer_file);
            out.close();
            out.u32(2);
            out.open();
            for (int child = 0; child < 2; ++child) {
                out.f32(0.f); // priority offset
                out.f32(1.f); // priority scale
            }
            out.close();
            out.flag(true);
            out.u32(1);
            out.open();
            write_level_geode(out, ids, all, level, stem);
            out.close();
            out.close();
        }

        void write_osgb_file(const std::vector<char>& scene, const std::filesystem::path& path) {
            const auto failed = [&] { return collisionError("mesh.write_failed", "Could not write the mesh; check disk space and permissions", ErrorCode::Unavailable, core::path_to_utf8(path)); };
            std::ofstream file(core::native_fs::native_path(path), std::ios::binary | std::ios::trunc);
            OsgbStream header;
            header.u32(0x6c910ea1u);
            header.u32(0x1afb4545u);
            header.u32(kSceneFile);
            header.u32(kOsgVersion);
            header.u32(kBinaryBrackets);
            header.text("zlib");
            file.write(header.data().data(), std::streamsize(header.data().size()));

            z_stream stream{};
            // 16 asks for gzip framing, which OSG's reader expects.
            if (deflateInit2(&stream, 6, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
                throw failed();
            stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(scene.data()));
            stream.avail_in = static_cast<uInt>(scene.size());
            std::vector<char> chunk(size_t{1} << 20);
            int result = Z_OK;
            do {
                stream.next_out = reinterpret_cast<Bytef*>(chunk.data());
                stream.avail_out = static_cast<uInt>(chunk.size());
                result = deflate(&stream, Z_FINISH);
                file.write(chunk.data(), std::streamsize(chunk.size() - stream.avail_out));
            } while (result == Z_OK);
            deflateEnd(&stream);
            file.close();
            if (result != Z_STREAM_END || !file)
                throw failed();
        }
    } // namespace

    void write_splat_mesh_osgb(const TiledMesh& tiled, const std::filesystem::path& folder, const core::ProgressCallback& progress, const std::stop_token stop) {
        const auto data = folder / "Data";
        core::native_fs::create_directories(data);
        {
            std::ofstream metadata(core::native_fs::native_path(folder / "metadata.xml"), std::ios::binary | std::ios::trunc);
            metadata << kMetadata;
            metadata.close();
            if (!metadata)
                throw collisionError("mesh.write_failed", "Could not write the mesh; check disk space and permissions", ErrorCode::Unavailable, core::path_to_utf8(folder));
        }
        const auto tile_name = [](const size_t tile) { return std::format("Tile_+{:03}_+{:03}", tile / 1000, tile % 1000); };
        for (size_t i = 0; i < tiled.levels.size(); ++i) {
            const auto& level = tiled.levels[i];
            core::reportProgress(stop, progress, float(i) / float(tiled.levels.size()), std::string(core::collisionStage("mesh.stage_write", "Writing model")));
            const auto name = tile_name(level.tile);
            const auto tile_folder = data / core::utf8_to_path(name);
            core::native_fs::create_directories(tile_folder);
            // The coarsest level opens the tile: DasViewer loads <tile>/<tile>.osgb. Each level holds the name of the next finer one.
            const auto file_of = [&](const size_t l, const size_t levels) {
                return l + 1 == levels ? name + ".osgb" : std::format("{}_L{}.osgb", name, levels - 1 - l);
            };
            size_t levels = 0;
            for (const auto& other : tiled.levels)
                levels += other.tile == level.tile;
            const auto file = file_of(level.level, levels);
            const auto stem = file.substr(0, file.size() - 5);
            OsgbStream scene;
            ObjectIds ids;
            if (level.level > 0)
                write_paged_lod(scene, ids, tiled.combined, level, stem, file_of(level.level - 1, levels));
            else
                write_level_geode(scene, ids, tiled.combined, level, stem);
            write_osgb_file(scene.data(), tile_folder / core::utf8_to_path(file));
        }
        LOG_INFO("Mesh: model written to {}", core::path_to_utf8(folder));
    }

} // namespace lfs::training
