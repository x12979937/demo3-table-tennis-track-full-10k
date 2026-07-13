import bpy
from pathlib import Path
base=Path(__file__).resolve().parent
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.gltf(filepath=str(base/'model.glb'))
# Apply transforms and join mesh objects for a simple visual asset export.
mesh_objs=[o for o in bpy.context.scene.objects if o.type=='MESH']
for o in mesh_objs:
    bpy.context.view_layer.objects.active=o
    o.select_set(True)
if mesh_objs:
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
# Blender 3.x/4.x export operator names differ.
out=str(base/'converted'/'table_tennis_rackets_ping_pong.obj')
try:
    bpy.ops.wm.obj_export(filepath=out, export_materials=True)
except Exception:
    bpy.ops.export_scene.obj(filepath=out, use_materials=True)
print('EXPORTED_OBJ', out, 'meshes', len(mesh_objs))
