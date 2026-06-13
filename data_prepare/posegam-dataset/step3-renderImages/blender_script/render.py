import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import random


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

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def assign_gpu():
    # Find gpu with most free memory
    import pynvml, random
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    selected_gpu = None
    available_gpus = []
    required_memory = 4194304000 

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        pci_bus_id = f"0000:{pci_info.bus:02x}:{pci_info.device:02x}"
        blender_gpu_id = f"CUDA_{gpu_name}_{pci_bus_id}"

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem_info.free 

        if free_memory > required_memory:
            available_gpus.append(blender_gpu_id)

    if len(available_gpus) > 0:
        selected_gpu = random.choice(available_gpus)
    
    pynvml.nvmlShutdown()
    return selected_gpu

def init_render(engine='CYCLES', resolution=512, geo_mode=False, force_cycles=False):
    # Force CYCLES engine if environment maps are used
    if force_cycles:
        engine = 'CYCLES'
        
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True

    if engine == 'CYCLES':
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
        bpy.context.scene.cycles.filter_type = 'BOX'
        bpy.context.scene.cycles.filter_width = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
        bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
        bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
        bpy.context.scene.cycles.use_denoising = True
            
        # bpy.context.preferences.addons['cycles'].preferences.get_devices()
        # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        
        cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
        cycles_preferences.refresh_devices()
        cycles_preferences.compute_device_type = "CUDA"  # or "OPENCL"

        assigned_gpu = assign_gpu()
        while assigned_gpu is None:
            assigned_gpu = assign_gpu()

        devices = cycles_preferences.devices
        for device in devices:
            if device.id == assigned_gpu:
                device.use = True
            else:
                device.use = False
    else:
        bpy.context.scene.eevee.taa_render_samples = 64
        bpy.context.scene.eevee.gi_diffuse_bounces = 0
    
def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False):
    if not any([save_depth, save_normal, save_albedo, save_mist]):
        return {}, {}
    outputs = {}
    spec_nodes = {}
    
    # Check render engine compatibility
    render_engine = bpy.context.scene.render.engine
    print(f"[INFO] Using render engine: {render_engine}")
    if render_engine == 'BLENDER_EEVEE' and save_mist:
        print("[WARNING] Mist pass may not work properly with EEVEE engine")
    
    bpy.context.scene.use_nodes = True
    # Use the active view layer instead of hardcoded name
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = save_depth
    view_layer.use_pass_normal = save_normal
    view_layer.use_pass_diffuse_color = save_albedo
    view_layer.use_pass_mist = save_mist
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        # final path is base_path + file_slots[0].path (if set file_slots[0].path to abs path, it will ignore the first slash, leading to wrong path)
        depth_file_output.base_path = '/'
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 10 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = '/'
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = '/'
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
        
    if save_mist:
        # Configure world mist settings with error handling
        try:
            world = bpy.data.worlds.get('World')
            if world is None:
                world = bpy.data.worlds.new('World')
                bpy.context.scene.world = world
            world.mist_settings.start = 0
            world.mist_settings.depth = 10
        except AttributeError:
            print("[WARNING] Mist settings not available in this render engine")
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = '/'
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
        
    return outputs, spec_nodes

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

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam, cam_empty

def init_lighting():
    # Clear any existing lighting setup
    clear_lighting_setup()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }

def clear_lighting_setup():
    """Clear existing lights and environment map setup.
    
    Returns:
        None
    """
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Clear world environment setup if it exists
    if bpy.context.scene.world is not None:
        world = bpy.context.scene.world
        if world.use_nodes and world.node_tree:
            # Clear all nodes in the world material
            world.node_tree.nodes.clear()
            # Disable nodes to ensure clean state
            world.use_nodes = False

def setup_environment_map(hdr_folder: str, strength: float = 1.0) -> bool:
    """Setup environment map lighting using HDR files from the specified folder.
    
    Args:
        hdr_folder: Path to folder containing HDR files
        strength: Strength/intensity of the environment lighting
        
    Returns:
        bool: True if setup successful, False otherwise
    """
    # Clear any existing lighting setup first
    clear_lighting_setup()
    
    # Find HDR files in the folder
    hdr_extensions = ['*.hdr']
    hdr_files = []
    
    for ext in hdr_extensions:
        hdr_files.extend(glob.glob(os.path.join(hdr_folder, ext)))
    
    if not hdr_files:
        print(f"[WARNING] No HDR files found in {hdr_folder}, falling back to manual lighting")
        return False
    
    # Randomly select an HDR file
    selected_hdr = random.choice(hdr_files)
    print(f"[INFO] Using environment map: {selected_hdr}")
    
    # Create or get the world material
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world
    
    # Enable nodes for the world material
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Add Environment Texture node
    env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
    env_tex_node.location = (-300, 0)
    
    # Load the HDR image
    try:
        hdr_image = bpy.data.images.load(selected_hdr)
        env_tex_node.image = hdr_image
    except Exception as e:
        print(f"[ERROR] Failed to load HDR file {selected_hdr}: {e}")
        return False
    
    # Add Background shader node
    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.location = (0, 0)
    background_node.inputs['Strength'].default_value = strength
    
    # Add World Output node
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    output_node.location = (300, 0)
    
    # Connect the nodes
    links.new(env_tex_node.outputs['Color'], background_node.inputs['Color'])
    links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    
    # Ensure film transparency is maintained (background will only contribute to lighting, not render)
    bpy.context.scene.render.film_transparent = True
    
    print(f"[INFO] Environment map setup complete with strength {strength}")
    return True


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
    # bpy.ops.object.mode_set(mode="OBJECT")
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
            
def delete_custom_normals():
     for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.view_layer.material_override = new_mat

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

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def debug_save_screen_coords(screen_coords: np.ndarray, half_width: float, half_height: float, 
                           filename: str, resolution: int = 512):
    """Save screen coordinates as a debug image to visualize vertex projections.
    
    Args:
        screen_coords: Array of screen coordinates (Nx2)
        half_width: Half of the camera's horizontal field of view
        half_height: Half of the camera's vertical field of view
        filename: Output filename for the debug image
        resolution: Image resolution for the debug output
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the camera's field of view boundary
        fov_x = [-half_width, half_width, half_width, -half_width, -half_width]
        fov_y = [-half_height, -half_height, half_height, half_height, -half_height]
        ax.plot(fov_x, fov_y, 'r-', linewidth=2, label='Camera FOV')
        
        # Plot the vertex projections
        if len(screen_coords) > 0:
            ax.scatter(screen_coords[:, 0], screen_coords[:, 1], 
                      c='blue', s=1, alpha=0.6, label=f'{len(screen_coords)} vertices')
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(-half_width * 1.5, half_width * 1.5)
        ax.set_ylim(-half_height * 1.5, half_height * 1.5)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Screen X')
        ax.set_ylabel('Screen Y')
        ax.set_title(f'Screen Coordinates Debug - {len(screen_coords)} vertices')
        ax.legend()
        
        # Save the image
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DEBUG] Saved screen coordinates debug image: {filename}")
        
    except ImportError:
        print("[DEBUG] matplotlib not available, skipping screen coordinates debug image")
    except Exception as e:
        print(f"[DEBUG] Error saving screen coordinates debug image: {e}")

def calculate_object_visibility(cam: bpy.types.Object, mesh_objects: List[bpy.types.Object], debug_path=None, debug=False) -> float:
    """Calculate the portion of object bounding box vertices visible in camera view using vectorized operations.
    
    Args:
        cam: Camera object
        mesh_objects: List of mesh objects to check visibility for
        
    Returns:
        float: Fraction of object vertices visible in camera (0.0 to 1.0)
    """
    if not mesh_objects:
        return 0.0
    
    # Get camera matrix
    camera_matrix = np.array(cam.matrix_world.inverted())
    
    # Get camera properties
    scene = bpy.context.scene
    render = scene.render
    camera_data = cam.data
    
    # Calculate camera projection parameters
    aspect_ratio = render.resolution_x / render.resolution_y
    # For perspective camera
    fov = camera_data.angle
    if aspect_ratio > 1:
        fov_x = fov
        fov_y = 2 * np.arctan(np.tan(fov / 2) / aspect_ratio)
    else:
        fov_y = fov
        fov_x = 2 * np.arctan(np.tan(fov / 2) * aspect_ratio)
    
    half_width = np.tan(fov_x / 2)
    half_height = np.tan(fov_y / 2)
    
    visible_vertices = 0
    total_vertices = 0
    
    for obj in mesh_objects:
        if obj.type == 'MESH' and obj.data.vertices:
            # Get all vertex coordinates at once using foreach_get (much faster)
            vertices = obj.data.vertices
            vertex_count = len(vertices)
            
            if vertex_count == 0:
                continue
                
            # Extract all vertex coordinates as numpy array (Nx3) using foreach_get
            vertex_coords = np.zeros((vertex_count, 3), dtype=np.float32)
            vertices.foreach_get("co", vertex_coords.ravel())
            vertex_coords = vertex_coords.reshape((vertex_count, 3))
            
            # Transform to world space using object matrix (vectorized)
            obj_matrix = np.array(obj.matrix_world)
            # Add homogeneous coordinate (Nx4)
            vertex_coords_homog = np.ones((vertex_count, 4), dtype=np.float32)
            vertex_coords_homog[:, :3] = vertex_coords
            
            # Transform all vertices to world space at once
            world_vertices = (obj_matrix @ vertex_coords_homog.T).T  # (4x4) @ (4xN) -> (4xN) -> (Nx4)
            
            # Transform all vertices to camera space at once
            cam_vertices = (camera_matrix @ world_vertices.T).T  # (4x4) @ (4xN) -> (4xN) -> (Nx4)
            
            # Check if any vertices are behind the camera (positive Z in camera space)
            # If so, this is an invalid camera position and we return 0 visibility
            behind_camera_mask = cam_vertices[:, 2] >= 0
            if np.any(behind_camera_mask):
                return 0.0  # Invalid camera position - mesh is partially/fully behind camera
            
            # Check which vertices are in front of camera (negative Z)
            in_front_mask = cam_vertices[:, 2] < 0
            
            # For vertices in front, project to screen space
            valid_z_mask = cam_vertices[:, 2] != 0
            combined_mask = in_front_mask & valid_z_mask
            
            if np.any(combined_mask):
                # Vectorized projection to screen space
                screen_coords = np.zeros((vertex_count, 2), dtype=np.float32)
                screen_coords[combined_mask, 0] = cam_vertices[combined_mask, 0] / -cam_vertices[combined_mask, 2]  # screen_x
                screen_coords[combined_mask, 1] = cam_vertices[combined_mask, 1] / -cam_vertices[combined_mask, 2]  # screen_y
                
                # Debug: Save screen coordinates as image
                if debug:
                    debug_save_screen_coords(screen_coords[combined_mask], half_width, half_height, 
                                        debug_path)
                
                # Check which vertices are within field of view (vectorized)
                in_fov_x = (screen_coords[:, 0] >= -half_width) & (screen_coords[:, 0] <= half_width)
                in_fov_y = (screen_coords[:, 1] >= -half_height) & (screen_coords[:, 1] <= half_height)
                in_fov_mask = in_fov_x & in_fov_y & combined_mask
                
                visible_vertices += np.sum(in_fov_mask)
            
            total_vertices += vertex_count
    
    return visible_vertices / total_vertices if total_vertices > 0 else 0.0

def generate_random_camera_position(base_location: Vector, translation_range: float, 
                                  bbox_size: float) -> Vector:
    """Generate a random camera position with translation.
    
    Args:
        base_location: Original camera location
        translation_range: Maximum translation as fraction of object size
        bbox_size: Size of object bounding box
        
    Returns:
        Vector: New camera position with random translation
    """
    # Calculate maximum translation distance
    max_translation = translation_range * bbox_size
    
    # Generate random translation in 3D
    translation = Vector((
        random.uniform(-max_translation, max_translation),
        random.uniform(-max_translation, max_translation),
        random.uniform(-max_translation, max_translation)
    ))
    
    return base_location + translation

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()
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
        if arg.export_textured_glb:                # Create copies of meshes for baking
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

    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode, force_cycles=arg.use_environment_map)
    outputs, spec_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist
    )
    
    # normalize scene
    # scale, offset = normalize_scene()
    # not change scene
    scale, offset = 1.0, Vector((0, 0, 0))
    print('[INFO] Scene normalized.')
    
    # Initialize camera and lighting
    cam, cam_target = init_camera()
    
    # Setup lighting - either environment map or manual lights
    if arg.use_environment_map:
        print('[INFO] Environment map lighting enabled - will resample for each view.')
        # We'll set up environment maps per view, so just ensure CYCLES engine is ready
    else:
        init_lighting()
        print('[INFO] Manual lighting initialized.')

    # Override material
    if arg.geo_mode:
        override_material()
    
    # Create a list of views
    cube_size = 1.05  # Size of the cube bounding box
    to_export = {
        "aabb": [[-cube_size, -cube_size, -cube_size], [cube_size, cube_size, cube_size]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    
    # Get mesh objects for visibility calculation
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    bbox_min, bbox_max = scene_bbox()
    bbox_size = max(bbox_max - bbox_min)
    
    views = json.loads(arg.views)
    for i, view in enumerate(views):
        # Calculate base camera position (original object-centric position)
        base_location = Vector((
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        ))

        # Set camera focal length
        cam.data.lens = 16 / np.tan(view['fov'] / 2)
        
        # Apply random translation if enabled
        if arg.random_camera_translation:
            attempts = 0
            found_valid_position = False
            
            # Try random positions first
            while attempts < arg.max_translation_attempts:
                # Generate random position
                test_position = generate_random_camera_position(
                    base_location, arg.translation_range, bbox_size
                )
                # Optionally translate target as well
                if arg.translate_target:
                    test_target = generate_random_camera_position(
                        Vector((0, 0, 0)), arg.translation_range, bbox_size
                    )
                else:
                    test_target = Vector((0, 0, 0))
                
                # Set camera and target positions temporarily to test visibility
                cam.location = test_position
                cam_target.location = test_target
                bpy.context.view_layer.update()
                
                # Calculate visibility
                visibility = calculate_object_visibility(cam, mesh_objects, debug_path=os.path.join(arg.output_folder, f'{i:03d}_debug.png'), debug=False)

                # Check if this position meets the threshold
                if visibility >= arg.visibility_threshold:
                    cam.location = test_position
                    cam_target.location = test_target
                    found_valid_position = True
                    print(f'[INFO] View {i}: Found valid random position with {visibility:.2f} visibility in {attempts + 1} attempts')
                    break
                
                attempts += 1
            
            # If no random position met the threshold, use original position
            if not found_valid_position:
                cam.location = base_location
                cam_target.location = Vector((0, 0, 0))
                print(f'[INFO] View {i}: No random position met visibility threshold, using original position')
        else:
            # Use original object-centric position
            cam.location = base_location
            cam_target.location = (0, 0, 0)
        
        # Re-sample environment map for each view if environment maps are enabled
        if arg.use_environment_map:
            env_success = setup_environment_map(arg.environment_map_folder, arg.environment_map_strength)
            if not env_success:
                print(f"[WARNING] View {i}: Environment map setup failed, falling back to manual lighting")
                init_lighting()
        
        if arg.save_depth:
            # Calculate actual distance from camera to object center after translation
            object_center = Vector((0, 0, 0))  # Object is centered at origin
            
            camera_to_center_distance = (cam.location - object_center).length
            # The object is in a cube of size ~2.1 (from -1.05 to 1.05), so diagonal is 1.05 * sqrt(3)
            object_extent = cube_size * np.sqrt(3)
            
            spec_nodes['depth_map'].inputs[1].default_value = camera_to_center_distance - object_extent
            spec_nodes['depth_map'].inputs[2].default_value = camera_to_center_distance + object_extent
        
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')
        for name, output in outputs.items():
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_{name}')
            
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}*.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
            
        # Save camera parameters
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }
        if arg.save_depth:
            # Calculate actual depth range based on camera position
            object_center = Vector((0, 0, 0))
            
            camera_to_center_distance = (cam.location - object_center).length
            object_extent = cube_size * np.sqrt(3)
            
            metadata['depth'] = {
                'min': camera_to_center_distance - object_extent,
                'max': camera_to_center_distance + object_extent
            }
        to_export["frames"].append(metadata)
    
    # Save the camera parameters
    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
        
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
        combined_texture: Optional combined texture to use for the final material
        
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
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
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
    parser.add_argument(
        '--random_camera_translation',
        action='store_true',
        help='Enable random camera translation (default: False).'
    )
    parser.add_argument(
        '--translation_range',
        type=float,
        default=1.0,
        help='Maximum translation distance as fraction of object size (default: 1.0).'
    )
    parser.add_argument(
        '--visibility_threshold',
        type=float,
        default=0.3,
        help='Minimum portion of object that must remain visible (default: 0.3).'
    )
    parser.add_argument(
        '--max_translation_attempts',
        type=int,
        default=10,
        help='Maximum attempts to find valid camera position (default: 10).'
    )
    parser.add_argument(
        '--translate_target',
        action='store_true',
        help='Also apply random translation to camera target (default: False).'
    )
    parser.add_argument(
        '--use_environment_map',
        action='store_true',
        help='Use HDR environment maps for lighting instead of manual lights. Forces CYCLES engine and maintains film transparency (default: False).'
    )
    parser.add_argument(
        '--environment_map_folder',
        type=str,
        default='/path/to/collected_hdr_files/',
        help='Path to folder containing HDR environment map files (.hdr, .exr, .hdri).'
    )
    parser.add_argument(
        '--environment_map_strength',
        type=float,
        default=1.0,
        help='Strength/intensity of the environment map lighting (default: 1.0).'
    )
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    