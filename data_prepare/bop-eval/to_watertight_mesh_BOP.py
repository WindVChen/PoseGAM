#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  code builder: Dora team (https://github.com/Seed3D/Dora)
from glob import glob
import shutil
import cubvh
import torch
import numpy as np
import trimesh
from diso import DiffMC, DiffDMC
import argparse
from tqdm import tqdm
import os
import json
import point_cloud_utils as pcu
import bpy
from typing import *

def cleanup_blender_resources() -> None:
    """Clean up Blender resources to free memory.
    
    This function removes all objects, materials, textures and images
    from the Blender scene to prevent memory leaks and resource buildup.
    
    Returns:
        None
    """
    # Remove all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    # Remove all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Remove all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)
    
    # Remove all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)
    
    # Remove all meshes that aren't used
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    
    # Perform garbage collection
    import gc
    gc.collect()
    
    print("[INFO] Cleaned up Blender resources")

def apply_smart_uv_unwrap(objects=None, margin: float = 0.005) -> None:
    """Apply smart UV unwrapping to specified mesh objects.
    
    Args:
        objects: List of objects to apply UV unwrapping to. If None, applies to all mesh objects.
        margin: Margin between UV islands
    
    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    
    if objects is None:
        objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    
    if not objects:
        print("[WARNING] No mesh objects found to unwrap.")
        return
    
    for obj in objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # Enter edit mode and select all faces
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        # Smart UV unwrap with specified settings
        bpy.ops.uv.smart_project(
            island_margin=margin,
            area_weight=.0,
            correct_aspect=True,
            angle_limit=66.0 * (3.14159265 / 180.0),  # Convert degrees to radians
        )
        
        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f'[INFO] Smart UV unwrapping applied to {len(objects)} mesh objects.')

def setup_bake_material(texture_resolution: int = 2048) -> Tuple[Dict[str, bpy.types.Image], bpy.types.Material]:
    """Create a new material for the copied objects with texture nodes for advanced baking.
    
    Args:
        texture_resolution: Resolution of the texture image
        
    Returns:
        Tuple[Dict[str, bpy.types.Image], bpy.types.Material]: Texture images and the created material
    """
    # Create images for diffuse baking passes
    bake_images = {
        'diffuse': bpy.data.images.new("BakedDiffuse", width=texture_resolution, height=texture_resolution),
    }
    
    # Set image properties
    for img_name, img in bake_images.items():
        img.file_format = 'PNG'
        img.alpha_mode = 'NONE'  # No alpha channel
    
    # Create a new material for baking
    bake_material = bpy.data.materials.new(name="BakeMaterial")
    bake_material.use_nodes = True
    nodes = bake_material.node_tree.nodes
    links = bake_material.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes for each texture
    tex_nodes = {}
    for img_name, img in bake_images.items():
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = img
        tex_node.name = f"{img_name}_texture"
        tex_nodes[img_name] = tex_node
    
    # Position the nodes
    tex_nodes['diffuse'].location = (-500, 400)
    
    # Add material output nodes
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 300)
    bsdf_node.inputs['Roughness'].default_value = 0.5  # Medium roughness
    
    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (300, 300)
    
    # Connect nodes
    links.new(tex_nodes['diffuse'].outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Note: We don't connect mix node yet since we'll use the textures individually for baking
    # The connections to mix node will be made during the baking process
    
    return bake_images, bake_material

def apply_material_to_objects(objects: List[bpy.types.Object], material: bpy.types.Material) -> None:
    """Apply a material to a list of objects.
    
    Args:
        objects: List of objects to apply material to
        material: Material to apply
        
    Returns:
        None
    """
    if not objects:
        return
        
    for obj in objects:
        # Clear existing materials
        if obj.data.materials:
            obj.data.materials.clear()
            
        # Assign new material
        obj.data.materials.append(material)


def bake_vertex_colors_to_texture(objects: List[bpy.types.Object], texture_resolution: int = 1024) -> None:
    """Bake vertex colors from PLY objects to texture maps.
    
    This function creates materials with Attribute nodes to read vertex colors,
    applies UV unwrapping, and bakes the vertex colors to texture images.
    
    Args:
        objects: List of objects with vertex colors to bake
        texture_resolution: Resolution of the texture to bake to
        
    Returns:
        None
    """
    if not objects:
        print("[WARNING] No objects provided for vertex color baking.")
        return
    
    # Set up render engine for baking
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.render.bake.margin = 16
    bpy.context.scene.render.bake.use_clear = True
    
    for obj in objects:
        print(f"[INFO] Baking vertex colors for object: {obj.name}")
        
        # Apply smart UV unwrapping first
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        apply_smart_uv_unwrap(objects=[obj], margin=0.0)
        
        # Create a new material for vertex color baking
        vertex_material = bpy.data.materials.new(name=f"VertexColorMaterial_{obj.name}")
        vertex_material.use_nodes = True
        nodes = vertex_material.node_tree.nodes
        links = vertex_material.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create Attribute node for vertex colors
        attr_node = nodes.new('ShaderNodeAttribute')
        attr_node.attribute_type = 'GEOMETRY'
        
        # Try to find the correct vertex color attribute name
        # Check if the mesh has color attributes
        color_attr_name = 'Col'  # Default
        if obj.data.color_attributes:
            # Use the first available color attribute
            color_attr_name = obj.data.color_attributes[0].name
            print(f"[INFO] Found vertex color attribute: {color_attr_name}")
        else:
            print(f"[INFO] No color attributes found, using default: {color_attr_name}")
        
        attr_node.attribute_name = color_attr_name
        attr_node.location = (-400, 300)
        
        # Create Image Texture node for baking target
        bake_image = bpy.data.images.new(f"VertexColorBaked_{obj.name}", 
                                        width=texture_resolution, height=texture_resolution)
        bake_image.file_format = 'PNG'
        bake_image.alpha_mode = 'NONE'
        
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = bake_image
        tex_node.name = "bake_target"
        tex_node.location = (-200, 300)
        
        # Create BSDF and Output nodes
        bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf_node.location = (0, 300)
        
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (300, 300)
        
        # Connect vertex colors to BSDF
        links.new(attr_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        # Apply the material to the object
        if obj.data.materials:
            obj.data.materials.clear()
        obj.data.materials.append(vertex_material)
        
        # Select the texture node for baking
        for node in nodes:
            node.select = node == tex_node
        nodes.active = tex_node
        
        # Select only this object for baking
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # Set bake settings for vertex colors
        bpy.context.scene.render.bake.use_selected_to_active = False
        bpy.context.scene.render.bake.use_pass_direct = False
        bpy.context.scene.render.bake.use_pass_indirect = False
        bpy.context.scene.render.bake.use_pass_color = True
        
        # Bake the vertex colors to texture
        print(f"[INFO] Baking vertex colors to texture for {obj.name}...")
        bpy.ops.object.bake(type='DIFFUSE')
        
        # Update material to use the baked texture instead of vertex colors
        # Remove the attribute node connection and connect the texture
        for link in bsdf_node.inputs['Base Color'].links:
            links.remove(link)
        links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        
        print(f"[INFO] Vertex color baking complete for {obj.name}")
    
    print(f"[INFO] Vertex color baking complete for {len(objects)} objects.")


def bake_texture_passes(textured_objects: List[bpy.types.Object], target_objects: List[bpy.types.Object], 
                      bake_images: Dict[str, bpy.types.Image], bake_material: bpy.types.Material, 
                      samples: int = 64) -> None:
    """Bake both diffuse and glossy texture passes from original objects to target objects.
    
    Args:
        textured_objects: List of source objects
        target_objects: List of target objects
        bake_images: Dictionary of images to bake to (diffuse, glossy, combined)
        bake_material: Material with image texture nodes
        samples: Number of render samples for baking
        
    Returns:
        None
    """
    if not textured_objects or not target_objects:
        print("[WARNING] Missing objects for baking.")
        return
    
    # Set up render engine for baking
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.render.bake.margin = 16
    bpy.context.scene.render.bake.use_clear = True
    
    # Common bake settings for all passes
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.context.scene.render.bake.cage_extrusion = 0.005
    
    # Get the nodes from the material
    nodes = bake_material.node_tree.nodes
    
    # Set up bake settings for diffuse
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True
    
    # Make diffuse the active node for baking
    diffuse_node = nodes.get("diffuse_texture")
    for node in nodes:
        node.select = node == diffuse_node
    nodes.active = diffuse_node
    
    # Select objects
    bpy.ops.object.select_all(action="DESELECT")
    for obj in target_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = target_objects[-1]
    for obj in textured_objects:
        obj.select_set(True)
    
    # Bake diffuse pass
    print('[INFO] Baking diffuse pass...')
    bpy.ops.object.bake(type='DIFFUSE')
    
    # Now update the material to use the combined texture for shading
    nodes = bake_material.node_tree.nodes
    links = bake_material.node_tree.links
    
    # Get all the nodes we need
    diffuse_tex = nodes.get("diffuse_texture")
    bsdf_node = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    
    # Clear any existing connections to the base color input of the BSDF
    for link in bsdf_node.inputs['Base Color'].links:
        links.remove(link)
    
    # Connect the combined texture to the BSDF
    links.new(diffuse_tex.outputs['Color'], bsdf_node.inputs['Base Color'])
      # Make combined texture the active one for viewing
    for node in nodes:
        node.select = node == diffuse_tex
    nodes.active = diffuse_tex

    print('[INFO] Texture baking complete.')

def export_textured_glb(filepath: str) -> None:
    """Export the scene as a textured GLB file.
    
    Args:
        filepath: Path to export the GLB file
        
    Returns:
        None
    """
    # Select all mesh objects
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    
    # Make sure we have at least one object selected
    if not any(obj.select_get() for obj in bpy.context.scene.objects):
        print("[WARNING] No mesh objects to export.")
        return
    
    # Export as GLB with materials and textures
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        use_selection=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_tangents=True,  # Include tangents for normal mapping
        export_attributes=True,
        export_cameras=False,
        export_lights=False,
        export_apply=True,  # Apply modifiers
        export_draco_mesh_compression_enable=False  # Disable Draco compression
    )
    
    print(f'[INFO] Textured mesh exported to {filepath}')
    
    # Clean up Blender resources after export
    cleanup_blender_resources()

def texture_mesh(textured_mesh_path: str, non_textured_mesh_path: str, output_folder: str = None) -> None:
    """Apply texturing to a mesh using Blender.
    
    Args:
        textured_mesh_path: Path to the mesh with textures to import (reference, .glb format)
        non_textured_mesh_path: Path to the mesh that will receive the baked textures (.obj format)
        output_folder: Output folder for the textured mesh, defaults to mesh's directory
        
    Returns:
        None
    """
    # Clear existing objects in the scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.wm.ply_import(filepath=textured_mesh_path)
    print(f'[INFO] Imported textured reference mesh from {textured_mesh_path}')
    
    # Get all imported textured objects
    textured_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    
    if not textured_objects:
        print(f'[ERROR] No mesh objects were imported from {textured_mesh_path}')
        return
    
    # Bake vertex colors to texture for PLY files (since PLY has vertex colors, not textures)
    print('[INFO] Baking vertex colors from PLY to texture...')
    bake_vertex_colors_to_texture(textured_objects, texture_resolution=1024)
    
    # Import the non-textured mesh (target for baking)
    if not non_textured_mesh_path.lower().endswith('.ply'):
        print(f'[WARNING] Expected PLY format for non-textured mesh: {non_textured_mesh_path}')
        return

    print(f'[INFO] Imported non-textured target mesh from {non_textured_mesh_path}')
    bpy.ops.wm.ply_import(filepath=non_textured_mesh_path)
    
    # Get the newly imported non-textured objects 
    # (they were added after the textured ones, so we can find them)
    all_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    nontextured_objects = [obj for obj in all_objects if obj not in textured_objects]
    
    if not nontextured_objects:
        print(f'[ERROR] No mesh objects were imported from {non_textured_mesh_path}')
        return
    
    # Apply UV unwrapping only to copied objects
    apply_smart_uv_unwrap(objects=nontextured_objects, margin=0.0)
    
    # Create and set up the advanced baking materials and textures
    bake_images, bake_material = setup_bake_material(texture_resolution=1024)
    
    # Apply baking material to the copies
    apply_material_to_objects(nontextured_objects, bake_material)
    
    # Bake the texture passes from original to copies
    bake_texture_passes(textured_objects, nontextured_objects, bake_images, bake_material, samples=64)
    
    # Delete original objects - no longer needed
    bpy.ops.object.select_all(action="DESELECT")
    for obj in textured_objects:
        obj.select_set(True)
    bpy.ops.object.delete()
    
    # Set default output folder if none provided
    if output_folder is None:
        output_folder = os.path.dirname(non_textured_mesh_path)
    
    # Export as GLB
    output_path = os.path.join(output_folder, 'mesh.glb')
    export_textured_glb(output_path)


def generate_dense_grid_points(
    bbox_min = np.array((-1.05, -1.05, -1.05)),#array([-1.05, -1.05, -1.05])
    bbox_max= np.array((1.05, 1.05, 1.05)),#array([1.05, 1.05, 1.05])
    resolution = 512,
    indexing = "ij"
):
    length = bbox_max - bbox_min#array([2.1, 2.1, 2.1])
    num_cells = resolution# 512
    x = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3) # e.g. 513*513*513, 3
    grid_size = [resolution + 1, resolution + 1, resolution + 1] # e.g. 513, 513, 513

    return xyz, grid_size


def remesh(grid_xyz, grid_size, mesh_path, remesh_path, resolution, use_pcu):
    eps = 2 / resolution
    mesh = trimesh.load(mesh_path, force='mesh')

    # normalize mesh to [-1,1]
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) / 2
    if np.any(center > 1e-4):
        raise ValueError(f"Mesh center {center} is not close to zero, cannot normalize mesh.")

    # if there is the bbmin and bbmax is the same in any dimension, we need raise an error
    if np.any(bbmax - bbmin == 0):
        raise ValueError(f"Bounding box min {bbmin} and max {bbmax} are the same in some dimension, cannot normalize mesh.")
    scale = 2.0 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    # save the scaled mesh
    bbox_min = np.array((-1.05, -1.05, -1.05))
    bbox_max= np.array((1.05, 1.05, 1.05))
    bbox_size = bbox_max - bbox_min

    if use_pcu:
        grid_sdf, fid, bc = pcu.signed_distance_to_mesh(grid_xyz, vertices.astype(np.float32), mesh.faces)
        grid_udf = torch.FloatTensor(np.abs(grid_sdf)).cuda().view((grid_size[0], grid_size[1], grid_size[2]))
    else:
        f = cubvh.cuBVH(torch.as_tensor(vertices, dtype=torch.float32, device='cuda'), torch.as_tensor(mesh.faces, dtype=torch.float32, device='cuda')) # build with numpy.ndarray/torch.Tensor
        grid_udf, _,_= f.unsigned_distance(grid_xyz, return_uvw=False)
        grid_udf = grid_udf.view((grid_size[0], grid_size[1], grid_size[2]))
    diffdmc = DiffDMC(dtype=torch.float32).cuda()
    vertices, faces = diffdmc(grid_udf, isovalue=eps, normalize= False)
    
    vertices = (vertices + 1) / grid_size[0] * bbox_size[0] + bbox_min[0]

    # reloc the mesh core to the origin (0,0,0)
    vertice_center = (vertices.max(0)[0] + vertices.min(0)[0]) / 2
    vertices = vertices - vertice_center

    mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())

    # keep the max component of the extracted mesh
    components = mesh.split(only_watertight=False)
    bbox = []
    for c in components:
        bbmin = c.vertices.min(0)
        bbmax = c.vertices.max(0)
        bbox.append((bbmax - bbmin).max())
    max_component = np.argmax(bbox)
    mesh = components[max_component]
    # !!!! Notice: this will export OBJ file. When import this to blender, blender will automatically apply a 90 degree rotation around X axis to the obj file (directly export this to obj file will automatically inverse the rotation, thus correct), so when then export to glb, the mesh geometry itself (list(trimesh.load('../mesh_scaled.glb', process=False).geometry.values())[0]) will be rotated 90 degree around X axis, while the glb file will be correct.
    mesh.export(remesh_path, encoding='ascii')

    mesh = trimesh.load(mesh_path, force='mesh')

    # normalize mesh to [-1,1]
    texture_vertices = mesh.vertices
    bbmin = texture_vertices.min(0)
    bbmax = texture_vertices.max(0)
    center = (bbmin + bbmax) / 2
    scale = 2.0 / (bbmax - bbmin).max()
    texture_vertices = (texture_vertices - center) * scale

    # align with the remeshed mesh
    texture_vertices = (texture_vertices - texture_vertices.min(0)) * (vertices.cpu().numpy().max(0) - vertices.cpu().numpy().min(0)) / (texture_vertices.max(0) - texture_vertices.min(0))  + vertices.cpu().numpy().min(0)

    mesh.vertices = texture_vertices.astype(np.float32)

    if 'tless' in mesh_path:
        # set vertice color to [0.4,0.4,0.4,1]
        vertex_colors = np.ones((len(mesh.vertices), 4), dtype=np.float32) * [0.4, 0.4, 0.4, 1.0]
        mesh.visual.vertex_colors = vertex_colors
    elif 'ycbv' in mesh_path:
        # YCBV meshes keep their UV + texture; just rotate from z-up to y-up and export.
        print(f"[INFO] Exporting YCBV textured mesh for {mesh_path}")

        # set vertice from z-up to y-up
        vertices = mesh.vertices
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 0, 1],
                                    [0, -1, 0]])
        rotated_vertices = vertices.dot(rotation_matrix.T)
        mesh.vertices = rotated_vertices.astype(np.float32)

        out_scale_name = 'mesh.glb'
        mesh.export(os.path.join(os.path.dirname(remesh_path), out_scale_name))

        return

    out_scale_name = 'mesh_scaled.ply'
    mesh.export(os.path.join(os.path.dirname(remesh_path), out_scale_name), encoding='ascii')
    
    # Extract output folder from remesh_path
    output_folder = os.path.dirname(remesh_path)
    
    # Use the original GLB as textured reference and the newly created OBJ as the target for texturing
    texture_mesh(textured_mesh_path=os.path.join(os.path.dirname(remesh_path), out_scale_name), non_textured_mesh_path=remesh_path, output_folder=output_folder)

    if os.path.exists(remesh_path):
        os.remove(remesh_path)

def main(resolution, meshes_paths, remesh_target_path, use_pcu, rank_id) -> None:
    grid_xyz,grid_size = generate_dense_grid_points(resolution = resolution)
    if use_pcu:
        grid_xyz = grid_xyz.astype(np.float32)
    else:
        grid_xyz = torch.FloatTensor(grid_xyz).cuda()

    for mesh_path in tqdm(meshes_paths, desc=f"Processing meshes (rank {rank_id})"):
        part_dir = remesh_target_path + '/' + os.path.basename(mesh_path.split('.')[0])
        os.makedirs(part_dir, exist_ok=True)

        # copy mesh_path to remesh_path
        shutil.copy(mesh_path, os.path.join(part_dir, 'origin.' + mesh_path.split('.')[-1]))

        basename = os.path.basename(mesh_path)
        remesh_path = part_dir + '/' + 'mesh.ply'

        print('process: '+remesh_path)
        remesh(grid_xyz, grid_size, mesh_path, remesh_path, resolution, use_pcu)
        torch.cuda.empty_cache()
    
    # Final cleanup of all Blender resources
    cleanup_blender_resources()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        default="128",
        type= int,
        help=".",
    )
    parser.add_argument(
        "--obj_dir_path",
        type= str,
        help="Directory containing the BOP CAD models (<dataset>/models/*.ply). The dataset "
             "name in this path selects the mesh handling: 'tless' -> gray vertex colors, "
             "'ycbv' -> keep original UV texture, otherwise -> remesh + bake texture.",
        default="/path/to/gigapose/gigaPose_datasets/datasets/ycbv/models"
    )
    parser.add_argument(
        "--remesh_target_path",
        type= str,
        help="Output directory for the watertight per-object meshes (<BOP_dir>/<dataset>/).",
        default="/ibex/tmp/TRELLIS-500K/BOP-data/ycbv/"
    )
    parser.add_argument(
        "--use_pcu",
        action='store_false',
        help="If set to False, use cubvh (GPU-required). \
            It's fast for meshes with a moderate number of faces \
            but becomes extremely slow or causes GPU memory leakage for large number of faces. \
            If set to True, use pcu (CPU-based).\
            It's generally slower than cubvh but avoids GPU leakage overflow issues.",
    )
    parser.add_argument("--rank_size", type=int, default=1, help="Number of processes to run in parallel. Default is 1, which means no parallel processing.")
    parser.add_argument("--rank_id", type=int, default=0, help="Rank ID for parallel processing. Default is 0, which means no parallel processing.")
    args, extras = parser.parse_known_args()

    meshes_paths = glob(os.path.join(args.obj_dir_path, '*.ply'))

    os.makedirs(args.remesh_target_path, exist_ok=True)

    # Filter paths based on rank size and ID
    if args.rank_size > 1:
        meshes_paths = [path for i, path in enumerate(meshes_paths) if i % args.rank_size == args.rank_id]
    else:
        meshes_paths = meshes_paths

    main(args.resolution, meshes_paths, args.remesh_target_path, args.use_pcu, args.rank_id)

