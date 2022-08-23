
import bpy
import pickle
import time
from os.path import join, splitext
import argparse
import sys

def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv

parser = argparse.ArgumentParser()
parser.add_argument('--shape', type=str)
argv = extract_args()
args = parser.parse_args(argv)
shape = args.shape

shape_dir = join("/home/evelyn/Desktop/partnet-reasoning/shapes/fix/close/fridge", shape)
filepath = join(shape_dir, "demo.pkl")

bpy.ops.wm.open_mainfile(filepath='./base_scene.blend')

# args
render_args = bpy.context.scene.render
# render_args.engine = 'CYCLES'
render_args.resolution_x = 256
render_args.resolution_y = 256
render_args.resolution_percentage = 100
render_args.tile_x = 256
render_args.tile_y = 256
cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
cycles_prefs.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'

time1 = time.time()

with open(filepath, 'rb') as pickle_file:
    data = pickle.load(pickle_file)

    for obj_key in data:
        pybullet_obj = data[obj_key]
        # Load mesh of each link
        assert pybullet_obj['type'] == 'mesh'

        # Delete lights and camera
        parts = 0
        final_objs = []

        for (i,mesh) in enumerate(pybullet_obj['mesh_path']):
            extension = splitext(mesh)[1]
        #extension = splitext(pybullet_obj['mesh_path'])[1]
        # Handle different mesh formats
            if 'obj' in extension:
                bpy.ops.import_scene.obj(
                    filepath=pybullet_obj['mesh_path'][i],
                    axis_forward='-Z', axis_up='Y')
            elif 'dae' in extension:
                bpy.ops.wm.collada_import(
                    filepath=pybullet_obj['mesh_path'][i])
            elif 'stl' in extension:
                bpy.ops.import_mesh.stl(
                    filepath=pybullet_obj['mesh_path'][i])
            else:
                print("Unsupported File Format:{}".format(extension))
        
            for import_obj in bpy.context.selected_objects:
                bpy.ops.object.select_all(action='DESELECT')
                import_obj.select = True
                if 'Camera' in import_obj.name \
                        or 'Light' in import_obj.name\
                        or 'Lamp' in import_obj.name:
                    bpy.ops.object.delete(use_global=True)
                else:
                    scale = pybullet_obj['mesh_scale']
                    if scale is not None:
                        import_obj.scale.x = scale[0]
                        import_obj.scale.y = scale[1]
                        import_obj.scale.z = scale[2]
                    final_objs.append(import_obj)
                    parts += 1

        bpy.ops.object.select_all(action='DESELECT')
        for obj in final_objs:
            if obj.type == 'MESH':
                obj.select = True
        if len(bpy.context.selected_objects):
            bpy.context.scene.objects.active =\
                bpy.context.selected_objects[0]
            # join them
            bpy.ops.object.join()
        blender_obj = bpy.context.scene.objects.active
        blender_obj.name = obj_key

        # Keyframe motion of imported object
        for frame_count, frame_data in enumerate(
                pybullet_obj['frames']):
            if frame_count % 1 != 0:
                continue
            pos = frame_data['position']
            orn = frame_data['orientation']
            bpy.context.scene.frame_set(
                frame_count // 1)
            # Apply position and rotation
            blender_obj.location.x = pos[0]
            blender_obj.location.y = pos[1]
            blender_obj.location.z = pos[2]
            blender_obj.rotation_mode = 'QUATERNION'
            blender_obj.rotation_quaternion.x = orn[0]
            blender_obj.rotation_quaternion.y = orn[1]
            blender_obj.rotation_quaternion.z = orn[2]
            blender_obj.rotation_quaternion.w = orn[3]

            bpy.context.object.keyframe_insert(
                'rotation_quaternion', group='Rotation')
            bpy.context.object.keyframe_insert(
                'location', group='Location')

# bpy.data.objects['Camera'].constraints['Track To'].target = bpy.data.objects[list(data.items())[0][0]]
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.data.scenes['Scene'].frame_end = 10
bpy.context.scene.render.filepath = join(shape_dir, "output", "pngs", "video")
bpy.ops.render.render(animation=True)

bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
# bpy.context.scene.render.filepath = join(shape_dir, "output", "depths", "depths")
# bpy.ops.render.render(animation=True)
render_node = bpy.context.scene.node_tree.nodes['Render Layers']

output_node2 = bpy.context.scene.node_tree.nodes.new('CompositorNodeOutputFile')
output_node2.base_path = join(shape_dir, "output", "depths")
output_node2.format.file_format = 'OPEN_EXR'
# output_node2.file_slots[0].path = join(shape_dir, "output", "depths", "depths")
link6 = bpy.context.scene.node_tree.links.new(render_node.outputs[2], output_node2.inputs[0])

bpy.ops.render.render(animation=True)


bpy.context.scene.render.image_settings.file_format = "PNG"
tree = bpy.context.scene.node_tree
render = tree.nodes['Render Layers']
links = tree.links

output_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeOutputFile')
output_node.base_path = join(shape_dir, "output", "masks")
i = 0

for obj in bpy.data.objects:
    # if name in obj.name:
    if obj.name == 'Camera' or 'Lamp' in obj.name or obj.name in ['Area', 'Empty', 'Ground']: continue

    i += 1
    obj.pass_index = i
    mask_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeIDMask')
    mask_node.index = i
    link = links.new(render.outputs["IndexOB"], mask_node.inputs["ID value"])
    output_node.layer_slots.new(str(i))
    link = links.new(mask_node.outputs[0], output_node.inputs[i])

# bpy.context.scene.render.filepath = join(shape_dir, "output", "masks", "masks")
bpy.ops.render.render(animation=True)


print (time.time() - time1)