import argparse, sys, os
from typing import *
import bpy
import numpy as np


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.wm.usd_import,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.wm.stl_import,
    "usda": bpy.ops.wm.usd_import,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.wm.ply_import,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)

def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)

def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)

def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")

def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)

    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
    print('[INFO] Scene initialized.')

    # Merge all meshes to one mesh by object joining
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    # Find the first mesh object to set as active
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if mesh_objects:
        bpy.context.view_layer.objects.active = mesh_objects[0]
    else:
        print("[WARNING] No mesh objects found to join.")
        return
    bpy.ops.object.join()
    print('[INFO] All meshes merged into one.')

    if arg.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        if arg.export_textured_glb:
            # Create copies of meshes for baking
            original_objects, copied_objects = copy_meshes_for_baking()

            # Apply UV unwrapping only to copied objects
            apply_smart_uv_unwrap(objects=copied_objects, margin=0.0)

            # Create and set up the advanced baking materials and textures
            bake_images, bake_material = setup_bake_material(texture_resolution=arg.texture_resolution)

            # Apply baking material to the copies
            apply_material_to_objects(copied_objects, bake_material)

            # Bake the texture passes from original to copies (diffuse and glossy)
            bake_texture_passes(original_objects, copied_objects, bake_images, bake_material, samples=arg.bake_samples)

            # Save the baked textures to disk
            diffuse_path = os.path.join(arg.output_folder, "baked_diffuse.png")
            glossy_path = os.path.join(arg.output_folder, "baked_glossy.png")
            transmission_path = os.path.join(arg.output_folder, "baked_transmission.png")
            combined_path = os.path.join(arg.output_folder, "baked_texture.png")

            bake_images['diffuse'].filepath_raw = diffuse_path
            bake_images['diffuse'].save()

            bake_images['glossy'].filepath_raw = glossy_path
            bake_images['glossy'].save()

            bake_images['transmission'].filepath_raw = transmission_path
            bake_images['transmission'].save()

            bake_images['combined'].filepath_raw = combined_path
            bake_images['combined'].save()

            print(f'[INFO] Saved baked textures to {arg.output_folder}')

            # Delete original objects - no longer needed
            bpy.ops.object.select_all(action="DESELECT")
            for obj in original_objects:
                obj.select_set(True)
            bpy.ops.object.delete()

            # Export as GLB
            export_textured_glb(os.path.join(arg.output_folder, 'mesh.glb'))

        # Also export as PLY mesh if requested
        if arg.export_ply:
            bpy.ops.wm.ply_export(filepath=os.path.join(arg.output_folder, 'mesh.ply'))
            print('[INFO] Also exported mesh as PLY.')

    return None

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
            angle_limit=66.0 * (3.14159265 / 180.0)
        )

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

    print(f'[INFO] Smart UV unwrapping applied to {len(objects)} mesh objects.')

def copy_meshes_for_baking() -> Tuple[List[bpy.types.Object], List[bpy.types.Object]]:
    """Make a copy of all mesh objects for texture baking.

    Returns:
        Tuple[List, List]: Original objects and their copies
    """
    bpy.ops.object.select_all(action="DESELECT")
    original_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    copied_objects = []

    if not original_objects:
        print("[WARNING] No mesh objects found to copy.")
        return [], []

    for obj in original_objects:
        # Select and make active
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Duplicate object
        bpy.ops.object.duplicate()
        copied_obj = bpy.context.active_object
        copied_obj.name = f"{obj.name}_bake_target"
        copied_objects.append(copied_obj)

    print(f'[INFO] Created {len(copied_objects)} mesh copies for baking.')
    return original_objects, copied_objects

def setup_bake_material(texture_resolution: int = 2048) -> Tuple[Dict[str, bpy.types.Image], bpy.types.Material]:
    """Create a new material for the copied objects with texture nodes for advanced baking.

    Args:
        texture_resolution: Resolution of the texture image

    Returns:
        Tuple[Dict[str, bpy.types.Image], bpy.types.Material]: Texture images and the created material
    """
    # Create images for diffuse and glossy baking passes
    bake_images = {
        'diffuse': bpy.data.images.new("BakedDiffuse", width=texture_resolution, height=texture_resolution),
        'glossy': bpy.data.images.new("BakedGlossy", width=texture_resolution, height=texture_resolution),
        'transmission': bpy.data.images.new("BakedTransmission", width=texture_resolution, height=texture_resolution),
        'combined': bpy.data.images.new("BakedCombined", width=texture_resolution, height=texture_resolution)
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
    tex_nodes['glossy'].location = (-500, 200)
    tex_nodes['transmission'].location = (-500, 0)
    tex_nodes['combined'].location = (-200, 300)

    # Add material output nodes
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 300)
    bsdf_node.inputs['Roughness'].default_value = 1  # Medium roughness

    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (300, 300)

    # Connect nodes
    links.new(tex_nodes['combined'].outputs['Color'], bsdf_node.inputs['Base Color'])
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


def bake_texture_passes(original_objects: List[bpy.types.Object], target_objects: List[bpy.types.Object],
                      bake_images: Dict[str, bpy.types.Image], bake_material: bpy.types.Material,
                      samples: int = 64) -> None:
    """Bake both diffuse and glossy texture passes from original objects to target objects.

    Args:
        original_objects: List of source objects
        target_objects: List of target objects
        bake_images: Dictionary of images to bake to (diffuse, glossy, combined)
        bake_material: Material with image texture nodes
        samples: Number of render samples for baking

    Returns:
        None
    """
    if not original_objects or not target_objects:
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

    # ---- STEP 1: BAKE DIFFUSE PASS ----
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
    for obj in original_objects:
        obj.select_set(True)

    # Bake diffuse pass
    print('[INFO] Baking diffuse pass...')
    bpy.ops.object.bake(type='DIFFUSE')

    # ---- STEP 2: BAKE GLOSSY PASS ----
    # Set up bake settings for glossy
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True

    # Make glossy the active node for baking
    glossy_node = nodes.get("glossy_texture")
    for node in nodes:
        node.select = node == glossy_node
    nodes.active = glossy_node

    # Re-select objects (may have been deselected)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in target_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = target_objects[-1]
    for obj in original_objects:
        obj.select_set(True)

    # Bake glossy pass
    print('[INFO] Baking glossy pass...')
    bpy.ops.object.bake(type='GLOSSY')

    # ---- STEP 3: BAKE TRANSMISSION PASS ----
    # Set up bake settings for transmission
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True

    # Make transmission the active node for baking
    transmission_node = nodes.get("transmission_texture")
    for node in nodes:
        node.select = node == transmission_node
    nodes.active = transmission_node
    # Re-select objects (may have been deselected)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in target_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = target_objects[-1]
    for obj in original_objects:
        obj.select_set(True)

    # Bake transmission pass
    print('[INFO] Baking transmission pass...')
    bpy.ops.object.bake(type='TRANSMISSION')

    # ---- STEP 4: COMBINE THE PASSES ----
    combine_baked_passes(bake_images, bake_material)

    print('[INFO] Texture baking complete.')

def combine_baked_passes(bake_images: Dict[str, bpy.types.Image], bake_material: bpy.types.Material) -> None:
    """Combine diffuse and glossy passes into a final texture.

    Args:
        bake_images: Dictionary of baked images
        bake_material: Material with texture nodes

    Returns:
        None
    """
    # Get the images from the dictionary
    diffuse_img = bake_images['diffuse']
    glossy_img = bake_images['glossy']
    transmission_img = bake_images['transmission']
    combined_img = bake_images['combined']
      # Get the pixel data as numpy arrays
    diffuse_pixels = np.array(diffuse_img.pixels[:])
    glossy_pixels = np.array(glossy_img.pixels[:])
    transmission_pixels = np.array(transmission_img.pixels[:])

    # Mix factor (1.0 = full glossy added to diffuse)
    mix_factor = 0.5

    # Reshape arrays for easier processing (separate RGBA channels)
    pixel_count = len(diffuse_pixels) // 4
    diffuse_reshaped = diffuse_pixels.reshape(pixel_count, 4)
    glossy_reshaped = glossy_pixels.reshape(pixel_count, 4)
    transmission_reshaped = transmission_pixels.reshape(pixel_count, 4)

    # Apply mixing with vectorized operations - much faster than pixel-by-pixel
    # Get RGB channels (first 3 columns)
    diffuse_rgb = diffuse_reshaped[:, :3]
    glossy_rgb = glossy_reshaped[:, :3] * mix_factor
    transmission_rgb = transmission_reshaped[:, :3] * mix_factor

    # Combine using numpy's clip to avoid values over 1.0
    combined_rgb = np.clip(diffuse_rgb + glossy_rgb, 0, 1.0)
    combined_rgb = np.clip(combined_rgb + transmission_rgb, 0, 1.0)

    # Create the combined image data (with alpha from diffuse)
    combined_reshaped = np.copy(diffuse_reshaped)
    combined_reshaped[:, :3] = combined_rgb

    # Flatten back to 1D array for Blender
    combined_pixels = combined_reshaped.flatten()

    # Update the combined image with the mixed pixels
    combined_img.pixels = combined_pixels

    print('[INFO] Diffuse and glossy passes combined successfully.')

    # Now update the material to use the combined texture for shading
    nodes = bake_material.node_tree.nodes
    links = bake_material.node_tree.links

    # Get all the nodes we need
    combined_tex = nodes.get("combined_texture")
    bsdf_node = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)

    # Clear any existing connections to the base color input of the BSDF
    for link in bsdf_node.inputs['Base Color'].links:
        links.remove(link)

    # Connect the combined texture to the BSDF
    links.new(combined_tex.outputs['Color'], bsdf_node.inputs['Base Color'])
      # Make combined texture the active one for viewing
    for node in nodes:
        node.select = node == combined_tex
    nodes.active = combined_tex

    # Delete the diffuse and glossy texture nodes as they are no longer needed
    diffuse_node = nodes.get("diffuse_texture")
    glossy_node = nodes.get("glossy_texture")
    transmission_node = nodes.get("transmission_texture")

    if diffuse_node:
        nodes.remove(diffuse_node)
    if glossy_node:
        nodes.remove(glossy_node)
    if transmission_node:
        nodes.remove(transmission_node)

    print('[INFO] Removed unused texture nodes to optimize material')

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
    # blender export will automatically include a 90 degree rotation
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
        export_apply=True  # Apply modifiers
    )

    print(f'[INFO] Textured mesh exported to {filepath}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads a 3D model and exports it as a (textured) mesh.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be processed.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh.')
    parser.add_argument(
        '--export_textured_glb',
        action='store_false',
        help='Export the mesh as a textured GLB file.'
    )
    parser.add_argument(
        '--export_ply',
        action='store_true',
        help='Export the mesh as PLY in addition to GLB.'
    )
    parser.add_argument(
        '--texture_resolution',
        type=int,
        default=1024,
        help='Resolution of the baked texture (default: 1024).'
    )
    parser.add_argument(
        '--bake_samples',
        type=int,
        default=64,
        help='Number of samples for texture baking (default: 64).'
    )
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
