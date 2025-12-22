import json
import argparse
from collections import defaultdict, OrderedDict
import codecs
import sys
import os
import shutil
from pathlib import Path
from py3dtiles.tileset.tileset import TileSet
from py3dtiles.typing import AssetDictType, GeometricErrorType, TilesetDictType
# coding=utf-8
from queue import Queue


class customTileSet(TileSet):
    def __init__(self, geometric_error: float = 500):
        super().__init__(geometric_error)
    
    @staticmethod
    def from_file(tileset_path: Path) -> TileSet:
        with tileset_path.open() as f:
            tileset_dict = json.load(f)

        tileset = customTileSet.from_dict(tileset_dict)
        tileset.root_uri = tileset_path.parent

        return tileset
          
    # 重写父类的方法，修复导出json时bouding volume不对问题
    def to_dict(self) -> TilesetDictType:
        """
        Convert to json string possibly mentioning used schemas
        """
        # Make sure the TileSet is aligned with its children Tiles.
        #self.root_tile.sync_bounding_volume_with_children()

        tileset_dict: TilesetDictType = {
            "root": self.root_tile.to_dict(),
            "asset": self.asset.to_dict(),
            "geometricError": self.geometric_error,
        }

        tileset_dict = self.add_root_properties_to_dict(tileset_dict)

        if self.extensions_used:
            tileset_dict["extensionsUsed"] = list(self.extensions_used)
        if self.extensions_required:
            tileset_dict["extensionsRequired"] = list(self.extensions_required)

        return tileset_dict
    
    
class GridData(object):
    def __init__(self, path, nodedata_epoch, imagery_epoch, metadata_epoch, bulk):
        self.path = path
        self.level = len(self.path)
        self.nodedata_epoch = nodedata_epoch
        self.metadata_epoch = metadata_epoch
        self.imagery_epoch = imagery_epoch
        self.bulk = bulk

def getGrids(gridsfile):
    grids = None

    try:
        f = codecs.open(gridsfile, 'r', 'utf-8')
        s = f.read()
        f.close()

        grids = json.loads(s)
    except Exception as e:
        print(e)
    return grids

def getGridOctants(grids):
    overlapping_octants = defaultdict(list)
    for level in range(1, len(grids)+1):
        nodes = grids[f'{level}']
        if len(nodes) > 0:
            for i in range(len(nodes)):
                n = nodes[i]
                grid = GridData(n['path'], n['nodedata_epoch'], n['imagery_epoch'], n['metadata_epoch'], n['bulk'])
                overlapping_octants[level].append(grid)
        else:
            overlapping_octants[level] = nodes

    return overlapping_octants

def parse_args():
    description = "usage: % prog[options]"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--dir', required=True, help='The directory for saving map 3dtiles')
    parser.add_argument('-g', '--grid', required=True, help='The grid config for map of selected area')
    parser.add_argument('-t', '--tileset', required=True, help='The tileset.json file of the 3dtiles')
    args = parser.parse_args()
    return args


# 递归方式实现深度优先遍历
def DFS(tile):
    queue = Queue()
    for child in tile.children:   
        if child.extras.__contains__('comment'):
            path = child.extras['comment'].split("=", 1)[1].strip("[ ']")
            # print(f"accessed:{path}")
            if path in path_list:
                #print(f"accessed:{path}")
                path_len = len(path)
                if path_len % 4 == 0 and path_len < 16:
                    if child.children[0].content_uri.suffix == ".json":
                        #sub_tileset_file = str(child.children[0].content_uri)
                        sub_tileset=customTileSet.from_file(Path(savedir + "/" + str(child.children[0].content_uri)))
                        DFS(sub_tileset.root_tile)
                        # save the modified json file
                        if not os.path.exists(Path(savedir + "/" + str(child.children[0].content_uri) + ".orig")):
                            shutil.copyfile(savedir + "/"+ str(child.children[0].content_uri), savedir+"/"+ str(child.children[0].content_uri)+".orig")
                        # write_to_json(sub_tileset, child.children[0].content_uri)
                        sub_tileset.write_as_json(Path(savedir+"/"+ str(child.children[0].content_uri)))
                elif path_len < 16:
                    DFS(child)
            else:
                queue.put(child)
    while not queue.empty():
        item = queue.get() # 出队
        path = item.extras['comment'].split("=", 1)[1].strip("[ ']")
        print(f"removed:{path}")
        tile.children.remove(item)


if __name__ == "__main__":
    # args = parse_args()
    gridsfile = "/mnt/sda/MapScape/3DTiles/chengxiaoya/switzerland_seq3/switzerland_seq3.json"
    savedir = "/mnt/sda/MapScape/3DTiles/chengxiaoya/switzerland_seq3"
    tileset_file = "/mnt/sda/MapScape/3DTiles/chengxiaoya/switzerland_seq3/v1/3dtiles/datasets/CgA/files/tileset.json"
    
    if tileset_file == '' :
        print('Please input the tileset file!')
        sys.exit(2) 
    if gridsfile == '':
        print('Please input the grid json file!')
        sys.exit(2)
    
    if savedir == '':
        print('Please input the model save directory!')
        sys.exit(2)

    if os.path.isfile(savedir):
        print('savedir can not be a file ', savedir)
        sys.exit(2)


    gridsjosn = getGrids(gridsfile)
    octants = getGridOctants(gridsjosn['grids'])
    path_list = []
    for level in sorted(octants):
        for octant in sorted(octants[level], key=lambda x: x.path):
            path_list.append(octant.path)

    #print(path_list)
    tileset = customTileSet.from_file(Path(tileset_file))
    DFS(tileset.root_tile)
    #save the top json file
    if not os.path.exists(str(tileset.root_uri)+"/tileset.json.orig"):
        shutil.copyfile(str(tileset.root_uri)+ "/tileset.json", str(tileset.root_uri)+"/tileset.json.orig")
    tileset.write_as_json(Path(tileset_file))           
    

    
