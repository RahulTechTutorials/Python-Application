import kivy
from kivy.app import App
from kivy.uix.button import Button

class CreateButtonApp(App):
    def build(self):
        return Button(text = 'Hello')
if __name__ == '__main__':
    %tb CreateButtonApp().run()
