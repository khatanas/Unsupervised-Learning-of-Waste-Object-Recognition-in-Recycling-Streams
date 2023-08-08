
channel_taxonomy = {
    # https://en.wikipedia.org/wiki/High-density_polyethylene
    'CH00004': {
        'supercategories': {
            #0: 'subparts',
            1: 'liquid container',
            2: 'conduct',
            3: 'transport solution',
            4: 'domestic furnitures',
        },

        'categories': {           
            #0: {
            #    'bottle cap': 1,                        # https://en.wikipedia.org/wiki/Bottle_cap
            #    'label': 2,                             # https://en.wikipedia.org/wiki/Label
            #    'lid': 3                                # https://en.wikipedia.org/wiki/Lid
            #},
            
            1: {
                'cistern': 1,                           # https://en.wikipedia.org/wiki/Cistern#/media/File:Kunststoffzisterne_IMG_20170714_120822174.jpg   
                'drum': 2,                              # https://en.wikipedia.org/wiki/Drum_(container)
                'jerrycan': 3,                          # https://en.wikipedia.org/wiki/Jerrycan
                'bucket': 4,                            # https://en.wikipedia.org/wiki/Bucket
                'bottle': 5                             # https://en.wikipedia.org/wiki/Bottle
            },
            
            2: {
                'HDPE pipe':1,                          # https://en.wikipedia.org/wiki/HDPE_pipe
                'plastic tubing':2                      # https://en.wikipedia.org/wiki/Tube_(fluid_conveyance)
            },
            
            3: {
                'plastic pallet': 1,                    # https://en.wikipedia.org/wiki/Pallet#Plastic_pallet
                'bottle crate': 2,                      # https://en.wikipedia.org/wiki/Bottle_crate
                'plastic basket': 3                     # https://en.wikipedia.org/wiki/Basket
            },
            
            4: {
                'waste container': 1                    # https://en.wikipedia.org/wiki/Waste_container#English
            }
        }
    },
    'XX00001': {
        'supercategories': {
            1: 'liquid container',
            2: 'conduct',
            3: 'transport solution',
            4: 'domestic furnitures',
        },
        'categories': {            
            1: {
                'cistern': 1,                           # https://en.wikipedia.org/wiki/Cistern#/media/File:Kunststoffzisterne_IMG_20170714_120822174.jpg   
                'drum': 2,                              # https://en.wikipedia.org/wiki/Drum_(container)
                'jerrycan': 3,                          # https://en.wikipedia.org/wiki/Jerrycan
                'bucket': 4,                            # https://en.wikipedia.org/wiki/Bucket
                'bottle': 5                             # https://en.wikipedia.org/wiki/Bottle
            },
            
            2: {
                'HDPE pipe':1,                          # https://en.wikipedia.org/wiki/HDPE_pipe
                'plastic tubing':2                      # https://en.wikipedia.org/wiki/Tube_(fluid_conveyance)
            },
            
            3: {
                'plastic pallet': 1,                    # https://en.wikipedia.org/wiki/Pallet#Plastic_pallet
                'bottle crate': 2,                      # https://en.wikipedia.org/wiki/Bottle_crate
                'plastic basket': 3                     # https://en.wikipedia.org/wiki/Basket
            },
            
            4: {
                'waste container': 1                    # https://en.wikipedia.org/wiki/Waste_container#English
            }
        }
    }
}


# reference images
channel_reference_ids = {
    'CH00004': {
        'empty': 'CH00004_20230317_095425',
        'sparse': 'CH00004_20230315_071910'
    },
    'XX00001': {
        'sparse': 'XX00001_20230318_000137'
    }
}


def getSuperCategories(channel):
    """
    Used to display {category_id: supercategory - category} when using AnnotationTool
    """
    taxonomy = channel_taxonomy[channel]
    supercategories = taxonomy['supercategories']
    categories = taxonomy['categories']
    
    category_ids = [int(str(sc)+str(categories[sc][obj])) for sc in supercategories for obj in categories[sc] if sc != 0]
    super_categories = [[supercategories[sc],obj] for sc in supercategories for obj in categories[sc]]
    
    return [[category_id,super_category] for category_id,super_category in zip(category_ids,super_categories)]



def getTaxonomy(channel):
    """
    Used to
    i) add to mask_classification file the raw used taxonomy
    ii) add to mask_classification file the generated text prompt from raw taxonomy
    iii) get easily the names of all objects belonging to taxonomy
    iv) get easily the names of subparts, to remove them from the image_classification procedure
    """
    
    # load 
    taxonomy = channel_taxonomy[channel]
    supercategories = taxonomy['supercategories']
    categories = taxonomy['categories']

    # generate CLIP text prompts
    text_prompt_categories = []
    for sc in supercategories:
        for c in categories[sc]:
            text_prompt_categories.append(
                f'part of a "{c}", on a black backround'
                )
    text_prompt_categories.append('it is not clear what the image shows')
    
    # used to assign names after CLIP classification
    names_all = [obj for sc in supercategories for obj in categories[sc]] + ['blur']
    # used to not collect images of subparts
    names_subparts = [obj for obj in categories[0]] + ['blur']
    
    return taxonomy, text_prompt_categories, names_all, names_subparts









#'supercategory':'fg',
#'id':1,
#'name':'fg'

# HDPE ==>                                      # https://en.wikipedia.org/wiki/High-density_polyethylene

#        'Caged IBC tote': 3                    # https://en.wikipedia.org/wiki/Caged_IBC_tote
#                'PET bottle': 4,                        # https://en.wikipedia.org/wiki/Bottle

            
'''            5: {
                'wall':1,                               # https://en.wikipedia.org/wiki/Wall
                'extruded aluminium profile':2,         # https://en.wikipedia.org/wiki/Extrusion
                'structural element': 3'''
