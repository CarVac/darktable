/*
    This file is part of darktable,
    copyright (c) 2009--2012 johannes hanika.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "iop/filmulate/filmSim.hpp"
#include <stdio.h>
using std::cout;
using std::endl;

extern "C"
{
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
// our includes go first:
#include "bauhaus/bauhaus.h"
#include "common/colorspaces.h"
#include "develop/imageop.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <stdlib.h>
#include <math.h>

// this is the version of the modules parameters,
// and includes version information about compile-time dt
DT_MODULE_INTROSPECTION(1, dt_iop_filmulate_params_t)

typedef struct dt_iop_filmulate_params_t
{
  // these are stored in db.
  // make sure everything is in here does not
  // depend on temporary memory (pointers etc)
  // stored in self->params and self->default_params
  // also, since this is stored in db, you should keep changes to this struct
  // to a minimum. if you have to change this struct, it will break
  // users data bases, and you should increment the version
  // of DT_MODULE(VERSION) above!
  float rolloff_boundary;
  float film_area;
  float layer_mix_const;
  int agitate_count;
} dt_iop_filmulate_params_t;

typedef struct dt_iop_filmulate_gui_data_t
{
  GtkWidget *rolloff_boundary, *film_area, *drama, *overdrive;
} dt_iop_filmulate_gui_data_t;

typedef struct dt_iop_filmulate_global_data_t
{
  // this is optionally stored in self->global_data
  // and can be used to alloc globally needed stuff
  // which is needed in gui mode and during processing.

} dt_iop_filmulate_global_data_t;

// this returns a translatable name
const char *name()
{
  // make sure you put all your translatable strings into _() !
  return _("filmulate");
}

// some additional flags (self explanatory i think):
int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

// where does it appear in the gui?
int groups()
{
  return IOP_GROUP_TONE;
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_filmulate_params_t *p = (dt_iop_filmulate_params_t *)p1;
  dt_iop_filmulate_params_t *d = (dt_iop_filmulate_params_t *)piece->data;

  d->rolloff_boundary = p->rolloff_boundary*65535.0f;
  d->film_area = powf(p->film_area,2.0f);
  d->layer_mix_const = p->layer_mix_const/100.0f;
  d->agitate_count = p->agitate_count;
}

/** modify regions of interest; filmulation requires the full image. **/
//The region of interest out is going to be the same as what it's given.
//Filmulator does not change this.
/*
void modify_roi_out(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece,
                    dt_iop_roi_t *roi_out, const dt_iop_roi_t *roi_in)
{
  *roi_out = *roi_in;
  cout << "modify roi out ==============================" << endl;
  cout << "roi in x: " << roi_in->x << endl;
  cout << "roi in y: " << roi_in->y << endl;
  cout << "roi in width: " << roi_in->width << endl;
  cout << "roi in height: " << roi_in->height << endl;
  cout << "roi in scale: " << roi_in->scale << endl;
  cout << "roi out x: " << roi_out->x << endl;
  cout << "roi out y: " << roi_out->y << endl;
  cout << "roi out width: " << roi_out->width << endl;
  cout << "roi out height: " << roi_out->height << endl;
  cout << "roi out scale: " << roi_out->scale << endl;
}
*/

// dt_iop_roi_t has 5 components: x, y, width, height, scale.
// The width and height are the viewport size.
//  When modifying roi_in, filmulator wants to change this to be the full image, scaled by the scale.
// The scale is the output relative to the input.
//===================NO IT ISN'T; we basically ignore it.
//  Filmulator doesn't want to change this.
// x and y are the viewport location relative to the full image area, at the viewport scale.
//  Filmulator wants to set this to 0.
void modify_roi_in(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *roi_out, dt_iop_roi_t *roi_in)
{
  /*
  cout << "modify roi in ==============================" << endl;
  cout << "roi in x: " << roi_in->x << endl;
  cout << "roi in y: " << roi_in->y << endl;
  cout << "roi in width: " << roi_in->width << endl;
  cout << "roi in height: " << roi_in->height << endl;
  cout << "roi in scale: " << roi_in->scale << endl;
  cout << "roi out x: " << roi_out->x << endl;
  cout << "roi out y: " << roi_out->y << endl;
  cout << "roi out width: " << roi_out->width << endl;
  cout << "roi out height: " << roi_out->height << endl;
  cout << "roi out scale: " << roi_out->scale << endl;
  */
  //roi_in->x = roi_out->x;
  //roi_in->y = roi_out->y;
  //roi_in->width = roi_out->width;
  //roi_in->height = roi_out->height;
  roi_in->scale = roi_out->scale;

  roi_in->x = 0;
  roi_in->y = 0;
  //cout << "unrounded width: " << piece->buf_in.width * roi_out->scale << endl;
  //cout << "unrounded height: " << piece->buf_in.height * roi_out->scale << endl;
  roi_in->width = round(piece->buf_in.width * roi_out->scale);
  roi_in->height = round(piece->buf_in.height * roi_out->scale);
  //roi_in->width = floor(piece->buf_in.width * roi_out->scale);
  //roi_in->height = floor(piece->buf_in.height * roi_out->scale);
  //cout << "roi in x: " << roi_in->x << endl;
  //cout << "roi in y: " << roi_in->y << endl;
  //cout << "roi in width: " << roi_in->width << endl;
  //cout << "roi in height: " << roi_in->height << endl;
  //cout << "roi in scale: " << roi_in->scale << endl;
  //*roi_in = *roi_out;
}

/** process, all real work is done here. */
void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const i, void *const o,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_color_intent_t intent = DT_INTENT_PERCEPTUAL;
  const cmsHPROFILE Lab = dt_colorspaces_get_profile(DT_COLORSPACE_LAB, "", DT_PROFILE_DIRECTION_ANY)->profile;
  const cmsHPROFILE Rec2020 = dt_colorspaces_get_profile(DT_COLORSPACE_LIN_REC2020, "", DT_PROFILE_DIRECTION_ANY)->profile;
  cmsHTRANSFORM transform_lab_to_lin_rec2020 = cmsCreateTransform(Lab, TYPE_LabA_FLT, Rec2020, TYPE_RGBA_FLT, intent, 0);
  cmsHTRANSFORM transform_lin_rec2020_to_lab = cmsCreateTransform(Rec2020, TYPE_RGBA_FLT, Lab, TYPE_LabA_FLT, intent, 0);

  const int width_in = roi_in->width;
  const int height_in = roi_in->height;
  const int x_out = roi_out->x;
  const int y_out = roi_out->y;
  const int width_out = roi_out->width;
  const int height_out = roi_out->height;

  //Temp buffer for the whole image
  float *rgbbufin = (float *)calloc(width_in * height_in * 4, sizeof(float));
  float *rgbbufout = (float *)calloc(width_out * height_out * 4, sizeof(float));

  //Turn Lab into linear Rec2020
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(rgbbufin, transform_lab_to_lin_rec2020)
#endif
  for(int y = 0; y < height_in; y++)
  {
    const float *in = (float*)i + y * width_in * 4;
    float *out = rgbbufin + y * width_in * 4;
    cmsDoTransform(transform_lab_to_lin_rec2020, in, out, width_in);
  }

  // this is called for preview and full pipe separately, each with its own pixelpipe piece.
  // Get the data struct.
  dt_iop_filmulate_params_t *d = (dt_iop_filmulate_params_t *)piece->data;
  float rolloff_boundary = d->rolloff_boundary;
  float film_area = d->film_area;
  float layer_mix_const = d->layer_mix_const;
  int agitate_count = d->agitate_count;

  //Filmulate things!
  filmulate(rgbbufin, rgbbufout,
            width_in, height_in,
            x_out, y_out,
            width_out, height_out,
            rolloff_boundary,
            film_area,
            layer_mix_const,
            agitate_count);

  free(rgbbufin);
  //Turn back to Lab
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(rgbbufout, transform_lin_rec2020_to_lab)
#endif
  for(int y = 0; y < height_out; y++)
  {
    const float *in = rgbbufout + y * width_out * 4;
    float *out = (float*)o + y * width_out * 4;
    cmsDoTransform(transform_lin_rec2020_to_lab, in, out, width_out);
  }
  free(rgbbufout);
}

/** optional: if this exists, it will be called to init new defaults if a new image is loaded from film strip
 * mode. */
void reload_defaults(dt_iop_module_t *module)
{
  // change default_enabled depending on type of image, or set new default_params even.

  // if this callback exists, it has to write default_params and default_enabled.
}

/** init, cleanup, commit to pipeline */
void init(dt_iop_module_t *module)
{
  // we don't need global data:
  module->data = malloc(sizeof(dt_iop_filmulate_global_data_t));
  module->params = calloc(1, sizeof(dt_iop_filmulate_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_filmulate_params_t));
  // our module is disabled by default
  // by default:
  module->default_enabled = 0;
  // order has to be changed by editing the dependencies in tools/iop_dependencies.py
  module->priority = 354; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_filmulate_params_t);
  module->gui_data = NULL;
  // init defaults:
  dt_iop_filmulate_params_t tmp = (dt_iop_filmulate_params_t){51275.0f/65535.0f, sqrtf(864.0f), 20.0f, 1};

  memcpy(module->params, &tmp, sizeof(dt_iop_filmulate_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_filmulate_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
  free(module->data);
  module->data = NULL;
}

/** put your local callbacks here, be sure to make them static so they won't be visible outside this file! */
static void rolloff_boundary_callback(GtkWidget *w, dt_iop_module_t *self)
{
  //This is important to avoid cycles!
  if(darktable.gui -> reset) 
  {
    return;
  }
  dt_iop_filmulate_params_t *p = (dt_iop_filmulate_params_t *)self->params;

  p->rolloff_boundary = dt_bauhaus_slider_get(w);
  //Let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  printf("rolloff_boundary callback 3\n");
}

//The slider goes from 0 to 65535, but we want to show 0 to 1.
static float rolloff_boundary_scaled_callback(GtkWidget *self, float input, dt_bauhaus_callback_t dir)
{
  float output;
  switch(dir)
  {
    case DT_BAUHAUS_SET:
      output = input*65535.0f;
      break;
    case DT_BAUHAUS_GET:
      output = input/65535.0f;
      break;
    default:
      output = input;
  }
  return output;
}

static void film_area_callback(GtkWidget *w, dt_iop_module_t *self)
{
  //This is important to avoid cycles!
  if(darktable.gui -> reset) 
  {
    return;
  }
  dt_iop_filmulate_params_t *p = (dt_iop_filmulate_params_t *)self->params;

  //The film area control is logarithmic WRT the linear dimensions of film.
  //But in the backend, it's actually using square millimeters of simulated film.
  //p->film_area = powf(expf(dt_bauhaus_slider_get(w)),2.0f);
  p->film_area = dt_bauhaus_slider_get(w);
  //Let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

//The film size slider displays the exponential of the linear slider position.
static float film_dimensions_callback(GtkWidget *self, float input, dt_bauhaus_callback_t dir)
{
  float output;
  switch(dir)
  {
    case DT_BAUHAUS_SET:
      //output = exp(input);
      output = log(fmax(input, 1e-15f));
      break;
    case DT_BAUHAUS_GET:
      //output = log(fmax(input, 1e-15f));
      output = exp(input);
      break;
    default:
      output = input;
  }
  return output;
}

static void drama_callback(GtkWidget *w, dt_iop_module_t *self)
{
  //This is important to avoid cycles!
  if(darktable.gui -> reset)
  {
    return;
  }
  dt_iop_filmulate_params_t *p = (dt_iop_filmulate_params_t *)self->params;

  //Drama goes from 0 to 100, but the relevant parameter in the backend is 0 to 1.
//  p->layer_mix_const = dt_bauhaus_slider_get(w)/100.0f;
  p->layer_mix_const = dt_bauhaus_slider_get(w);
  cout << "layer mix const at callback: " << p->layer_mix_const << endl;
  cout << "layer mix const at callback2: " << dt_bauhaus_slider_get(w) << endl;
  //Let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

//The slider goes from 0 to 1, but we want to show 0 to 100.
static float drama_scaled_callback(GtkWidget *self, float input, dt_bauhaus_callback_t dir)
{
  float output;
  switch(dir)
  {
    case DT_BAUHAUS_SET:
      output = input/100.0f;
      break;
    case DT_BAUHAUS_GET:
      output = input*100.0f;
      break;
    default:
      output = input;
  }
  return output;
}

static void overdrive_callback(GtkWidget *w, dt_iop_module_t *self)
{
  //This is important to avoid cycles!
  if(darktable.gui->reset)
  {
    return;
  }
  dt_iop_filmulate_params_t *p = (dt_iop_filmulate_params_t *)self->params;

  //If overdrive is off, we agitate once. If overdrive is on, we don't agitate.
  p->agitate_count = (dt_bauhaus_combobox_get(w) == 0) ? 1 : 0;
  //Let core know of the changes
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

/** gui callbacks, these are needed. */
void gui_update(dt_iop_module_t *self)
{
  // let gui slider match current parameters:
  dt_iop_filmulate_gui_data_t *g = (dt_iop_filmulate_gui_data_t *)self->gui_data;
  dt_iop_filmulate_params_t *p = (dt_iop_filmulate_params_t *)self->params;
  dt_bauhaus_slider_set(g->rolloff_boundary, p->rolloff_boundary);
  dt_bauhaus_slider_set(g->film_area, p->film_area);
  dt_bauhaus_slider_set(g->drama, p->layer_mix_const);
  dt_bauhaus_combobox_set(g->overdrive, (p->agitate_count == 0) ? 1 : 0);
}

void gui_init(dt_iop_module_t *self)
{
  // init the slider (more sophisticated layouts are possible with gtk tables and boxes):
  self->gui_data = malloc(sizeof(dt_iop_filmulate_gui_data_t));
  dt_iop_filmulate_gui_data_t *g = (dt_iop_filmulate_gui_data_t *)self->gui_data;

  //Create the widgets
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  g->rolloff_boundary = dt_bauhaus_slider_new_with_range(self, 1.0f, 65535.0f, 0, 51275.0f, 2);
  g->film_area = dt_bauhaus_slider_new_with_range(self, 1.2f, 6.0f, 0.001f, logf(sqrtf(864.0f)), 2);
  g->drama = dt_bauhaus_slider_new_with_range(self, 0.0f, 1.0f, 0.0f, 0.2f, 2);
  g->overdrive = dt_bauhaus_combobox_new(self);

  dt_bauhaus_slider_set_callback(g->rolloff_boundary, rolloff_boundary_scaled_callback);
  dt_bauhaus_slider_set_callback(g->film_area, film_dimensions_callback);
  dt_bauhaus_slider_set_callback(g->drama, drama_scaled_callback);
  dt_bauhaus_combobox_add(g->overdrive, _("off"));
  dt_bauhaus_combobox_add(g->overdrive, _("on"));
  
  dt_bauhaus_widget_set_label(g->rolloff_boundary, NULL, _("rolloff boundary"));
  gtk_widget_set_tooltip_text(g->rolloff_boundary, _("sets the point above which the highlights gently stop getting brighter. if you've got completely unclipped highlights before filmulation, raise this to 1."));
  dt_bauhaus_widget_set_label(g->film_area, NULL, _("film size"));
  gtk_widget_set_tooltip_text(g->film_area, _("larger sizes emphasize smaller details and overall flatten the image. smaller sizes emphasize larger regional contrasts. don't use larger sizes with high drama or you'll get the HDR look."));
  dt_bauhaus_widget_set_label(g->drama, NULL, _("drama"));
  gtk_widget_set_tooltip_text(g->drama, _("pulls down highlights to retain detail. this is the real \"filmy\" effect. this not only helps bring down highlights, but can rescue extremely saturated regions such as flowers."));
  dt_bauhaus_widget_set_label(g->overdrive, NULL, _("overdrive mode"));
  gtk_widget_set_tooltip_text(g->overdrive, _("in case of emergency, break glass and press this button. this increases the filminess, in case 100 Drama was not enough for you."));

  //Add widgets to the gui
  gtk_box_pack_start(GTK_BOX(self->widget), g->rolloff_boundary, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->film_area, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->drama, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->overdrive, TRUE, TRUE, 0);

  //Connect to the signals when widgets are changed
  g_signal_connect(G_OBJECT(g->rolloff_boundary), "value-changed", G_CALLBACK(rolloff_boundary_callback), self);
  g_signal_connect(G_OBJECT(g->film_area), "value-changed", G_CALLBACK(film_area_callback), self);
  g_signal_connect(G_OBJECT(g->drama), "value-changed", G_CALLBACK(drama_callback), self);
  g_signal_connect(G_OBJECT(g->overdrive), "value-changed", G_CALLBACK(overdrive_callback), self);
}

void gui_cleanup(dt_iop_module_t *self)
{
  // nothing else necessary, gtk will clean up the slider.
  free(self->gui_data);
  self->gui_data = NULL;
}

}//extern "C"

/** additional, optional callbacks to capture darkroom center events. */
// void gui_post_expose(dt_iop_module_t *self, cairo_t *cr, int32_t width, int32_t height, int32_t pointerx,
// int32_t pointery);
// int mouse_moved(dt_iop_module_t *self, double x, double y, double pressure, int which);
// int button_pressed(dt_iop_module_t *self, double x, double y, double pressure, int which, int type,
// uint32_t state);
// int button_released(struct dt_iop_module_t *self, double x, double y, int which, uint32_t state);
// int scrolled(dt_iop_module_t *self, double x, double y, int up, uint32_t state);

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-space on;