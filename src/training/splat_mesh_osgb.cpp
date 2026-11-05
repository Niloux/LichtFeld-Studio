/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_volume.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "splat_mesh.hpp"
#include <cfloat>
#include <format>
#include <fstream>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/PagedLOD>
#include <osg/Texture2D>
#include <osgDB/Options>
#include <osgDB/Registry>
#include <stb_image_write.h>

namespace lfs::training {
    namespace {
        using core::collisionError;

        constexpr int kOsgbJpegQuality = 95;

        // DasViewer's own model layout. The model is in the dataset's local coordinates; this origin only gives it a place.
        constexpr const char* kMetadata = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
                                          "<ModelMetadata version=\"1\">\n"
                                          "\t<SRS>ENU:31.0,121.0</SRS>\n"
                                          "\t<SRSOrigin>0,0,0</SRSOrigin>\n"
                                          "\t<Texture>\n\t\t<ColorSource>Visible</ColorSource>\n\t</Texture>\n"
                                          "</ModelMetadata>\n";

        void write_jpeg_file(const std::vector<uint8_t>& rgb, const int side, const std::filesystem::path& path) {
            std::ofstream out(core::native_fs::native_path(path), std::ios::binary | std::ios::trunc);
            const auto append = [](void* context, void* data, const int size) { static_cast<std::ofstream*>(context)->write(static_cast<const char*>(data), size); };
            if (!stbi_write_jpg_to_func(append, &out, side, side, 3, rgb.data(), kOsgbJpegQuality))
                throw collisionError("mesh.write_failed", "Could not write the mesh texture", ErrorCode::Unavailable, core::path_to_utf8(path));
            out.close();
            if (!out)
                throw collisionError("mesh.write_failed", "Could not write the mesh; check disk space and permissions", ErrorCode::Unavailable, core::path_to_utf8(path));
        }

        // The faces of one level, one drawable per page. The image carries only a name: the writer embeds the JPEG of that name.
        [[nodiscard]] osg::ref_ptr<osg::Geode> build_level(const SplatMesh& all, const TileLevel& level, const std::filesystem::path& folder, const std::string& stem,
                                                           std::vector<std::filesystem::path>& temporary) {
            osg::ref_ptr<osg::Geode> geode = new osg::Geode;
            // Only the vertices of this level, shared by all its pages.
            osg::ref_ptr<osg::Vec3Array> positions = new osg::Vec3Array(unsigned(level.vertex_count));
            osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array(unsigned(level.vertex_count));
            osg::ref_ptr<osg::Vec2Array> coordinates = new osg::Vec2Array(unsigned(level.vertex_count));
            for (size_t v = 0; v < level.vertex_count; ++v) {
                const auto& p = all.vertices[level.first_vertex + v];
                const auto& n = all.normals[level.first_vertex + v];
                const auto& uv = all.uvs[level.first_vertex + v];
                (*positions)[unsigned(v)].set(p[0], p[1], p[2]);
                (*normals)[unsigned(v)].set(n[0], n[1], n[2]);
                // A JPEG's first row is the top of the texture, which OSG counts from the bottom.
                (*coordinates)[unsigned(v)].set(uv[0], 1.f - uv[1]);
            }
            for (size_t p = 0; p < level.page_count; ++p) {
                const size_t page = level.first_page + p;
                const int side = all.page_side(page);
                const auto name = std::format("{}_{}.jpg", stem, p);
                const auto path = folder / core::utf8_to_path(name);
                write_jpeg_file(all.pages[page], side, path);
                temporary.push_back(path);

                osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
                geometry->setUseDisplayList(false);
                geometry->setUseVertexBufferObjects(true);
                geometry->setVertexArray(positions);
                geometry->setNormalArray(normals, osg::Array::BIND_PER_VERTEX);
                geometry->setTexCoordArray(0, coordinates);
                const size_t first = all.page_first_face[page], last = all.page_first_face[page + 1];
                osg::ref_ptr<osg::DrawElementsUInt> elements = new osg::DrawElementsUInt(GL_TRIANGLES, unsigned((last - first) * 3));
                for (size_t f = first; f < last; ++f)
                    for (int c = 0; c < 3; ++c)
                        (*elements)[unsigned((f - first) * 3 + size_t(c))] = unsigned(size_t(all.faces[f][size_t(c)]) - level.first_vertex);
                geometry->addPrimitiveSet(elements);

                osg::ref_ptr<osg::Image> image = new osg::Image;
                image->setFileName(name);
                image->setImage(side, side, 1, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, const_cast<unsigned char*>(all.pages[page].data()), osg::Image::NO_DELETE);
                osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D(image);
                texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
                texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
                texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
                texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
                // The photos already hold the light of the scene.
                auto* state = geometry->getOrCreateStateSet();
                state->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
                state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
                geode->addDrawable(geometry);
            }
            return geode;
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
            std::vector<std::filesystem::path> temporary;
            const auto stem = file_of(level.level, levels);
            auto geode = build_level(tiled.combined, level, tile_folder, stem.substr(0, stem.size() - 5), temporary);
            osg::ref_ptr<osg::Node> root = geode;
            if (level.level > 0) {
                // This level shows until the tile spans as many pixels as this level has texels.
                const float full_detail = 2.f * level.radius / level.texel;
                osg::ref_ptr<osg::PagedLOD> paged = new osg::PagedLOD;
                paged->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
                paged->setCenter(osg::Vec3(level.center[0], level.center[1], level.center[2]));
                paged->setRadius(level.radius);
                paged->setRangeMode(osg::LOD::PIXEL_SIZE_ON_SCREEN);
                paged->addChild(geode, 0.f, full_detail);
                paged->setFileName(1, file_of(level.level - 1, levels));
                paged->setRange(1, full_detail, FLT_MAX);
                root = paged;
            }
            osg::ref_ptr<osgDB::Options> options = new osgDB::Options("WriteImageHint=IncludeFile Compressor=zlib");
            options->setDatabasePath(core::path_to_utf8(tile_folder));
            const auto path = tile_folder / core::utf8_to_path(stem);
            const auto written = osgDB::Registry::instance()->writeNode(*root, core::path_to_utf8(path), options);
            std::error_code error;
            for (const auto& file : temporary)
                std::filesystem::remove(core::native_fs::native_path(file), error);
            if (!written.success())
                throw collisionError("mesh.write_failed", "Could not write the mesh model", ErrorCode::Unavailable, std::format("{}: {}", core::path_to_utf8(path), written.message()));
        }
        LOG_INFO("Mesh: model written to {}", core::path_to_utf8(folder));
    }

} // namespace lfs::training
