from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.core.window import Window

class MyApp(App):
    def build(self):
        #Window.size = (800, 600)
        main_layout = BoxLayout(orientation='vertical')

        grid_layout = GridLayout(cols=3, size_hint_y=None, height=1500)

        # Left image and button
        self.img1 = Image(source='cesta k souboru', size=(300, 300))
        import_btn = Button(text='Vložit obrázek')
        import_btn.bind(on_press=self.import_image1)

        img1_layout = BoxLayout(orientation='vertical')
        img1_layout.add_widget(self.img1)
        img1_layout.add_widget(import_btn)
        grid_layout.add_widget(img1_layout)

        # Convert button
        convert_btn = Button(text='Convert')
        convert_btn.bind(on_press=self.convert_image)
        grid_layout.add_widget(convert_btn)

        # Right image and download button
        self.img2 = Image(source='path_to_default_image2.png')
        download_btn = Button(text='Download Image')
        download_btn.bind(on_press=self.download_image)

        img2_layout = BoxLayout(orientation='vertical')
        img2_layout.add_widget(self.img2)
        img2_layout.add_widget(download_btn)
        grid_layout.add_widget(img2_layout)

        main_layout.add_widget(grid_layout)

        return main_layout

    def import_image1(self, instance):
        self.show_file_chooser(self.img1)

    def convert_image(self, instance):
        # Perform conversion logic here
        pass

    def download_image(self, instance):
        # Perform download logic here
        pass

    def show_file_chooser(self, image_widget):
        content = FileChooserIconView()
        popup = Popup(title='Choose an image file', content=content, size_hint=(0.9, 0.9))
        content.bind(on_submit=lambda x, selection, touch: self.load_image(image_widget, selection, popup))
        popup.open()

    def load_image(self, image_widget, selection, popup):
        if selection:
            image_widget.source = selection[0]
        popup.dismiss()

if __name__ == '__main__':
    MyApp().run()
