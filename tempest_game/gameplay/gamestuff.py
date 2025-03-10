import pygame

import config
import main
import gameplay.player2d as player2d
import gameplay.levels as levels
import rendering.neon as neon
import rendering.threedee as threedee
import rendering.levelbuilder3d as levelbuilder3d
import keybinds
import util.utility_functions as utility_functions
import util.fonts as fonts
import gameplay.highscores as highscores
from sound_manager.SoundManager import SoundManager


class GameplayMode(main.GameMode):

    def __init__(self, loop, gesture_detector):
        super().__init__(loop)
        self.player = player2d.Player()
        self.current_level = levels.InfiniteGeneratingLevel(9)

        self.camera_min_y = -1  # camera y when player is grounded
        self.camera_max_y = 1   # camera y when player is at max jump height
        self.camera_move_speed_pcnt = 6

        self.camera = threedee.Camera3D()

        self.camera.position.y = self.camera_min_y
        self.camera_z_offset = -44
        self.unload_offset = -30

        self.rotation_speed = 4
        self.current_rotation = 0

        self.foresight = 150
        self.neon_renderer = neon.NeonRenderer()

        self.score_font = fonts.get_font(30, name="cool")
        self.update_level_rotation(1000, snap=True)

        self.gesture_detector = gesture_detector
    
    # 依手勢辨識結果執行動作
    def handle_gesture_actions(self, gesture_actions):
        slide_detected = False  # 滑行型態
        if gesture_actions:
            for action in gesture_actions:
                if action == "jump":
                    self.player.jump()
                elif action == "left":
                    self.player.move_left()
                elif action == "right":
                    self.player.move_right()
                elif action == "slide":
                    self.player.slide()
                    slide_detected = True  # 偵測到滑行手勢

            # 本次手勢偵測不為 slide 動作，恢復為跑步型態
            if not slide_detected and self.player.is_sliding():
                self.player.run()

    def on_mode_start(self):
        SoundManager.play_song('game_theme', fadeout_ms=250, fadein_ms=1000)

    def update(self, dt, events, gesture_actions=None):
        self.handle_events(events)
        # 處理手勢動作
        self.handle_gesture_actions(gesture_actions)
        self.player.update(dt, self.current_level, events)

        self.update_camera_position(dt)
        self.update_level_rotation(dt)

        self.current_level.unload_obstacles(self.camera.position.z + self.unload_offset)

        if self.player.is_dead():
            score = self.player.get_score()
            highscores.add_new_score(score)
            self.loop.set_mode(RetryMenu(self.loop, score, self.player.get_death_message(), self.gesture_detector, self))

    def handle_events(self, events):
        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key in keybinds.MENU_CANCEL:
                    self.loop.set_mode(PauseMenu(self.loop, self, self.gesture_detector))
                if e.key in keybinds.RESET:
                    self.loop.set_mode(GameplayMode(self.loop, self.gesture_detector))

    def update_camera_position(self, dt):
        self.camera.position.z = self.player.z + self.camera_z_offset

        if self.player.y <= 0 or not config.Display.camera_bob:
            ideal_y = self.camera_min_y
        else:
            ideal_y = utility_functions.lerp(min(1, self.player.y / self.player.max_jump_height()),
                                             self.camera_min_y,
                                             self.camera_max_y)
        if abs(self.camera.position.y - ideal_y) < 0.01:
            self.camera.position.y = ideal_y
        else:
            dist = ideal_y - self.camera.position.y
            self.camera.position.y += dist * self.camera_move_speed_pcnt * dt

    def update_level_rotation(self, dt, snap=False):
        z = self.player.z
        ideal_rotation = levelbuilder3d.get_rotation_to_make_lane_at_bottom(z, self.player.lane, self.current_level)
        cur_rotation = self.current_level.get_rotation(z)
        dist, clockwise = utility_functions.abs_angle_between_angles(cur_rotation, ideal_rotation)
        if dist < 0.01 or snap:
            self.current_level.set_rotation(ideal_rotation)
        else:
            target_rots = [ideal_rotation - 360, ideal_rotation, ideal_rotation + 360]
            potential_rotations = [(t - cur_rotation) * self.rotation_speed * dt for t in target_rots]
            change_in_rotation = min(potential_rotations, key=abs)
            if abs(change_in_rotation) > dist:
                self.current_level.set_rotation(ideal_rotation)
            else:
                self.current_level.set_rotation(cur_rotation + change_in_rotation)

    def draw_to_screen(self, screen, extra_darkness_factor=1, show_score=True):
        screen.fill((0, 0, 0))
        all_lines = []
        cell_length = self.current_level.get_cell_length()
        z = self.camera.position.z
        n_lanes = self.current_level.number_of_lanes()
        cell_start = int(z / cell_length)
        cell_end = int((z + self.foresight) / cell_length + 1)

        for i in range(cell_start, cell_end):
            all_lines.extend(levelbuilder3d.build_section(i * cell_length, cell_length, self.current_level))

        for n in range(n_lanes):
            obstacles = self.current_level.get_all_obstacles_between(n, z, z + self.foresight)
            for obs in reversed(obstacles):
                # add them from from back to front so they overlap properly
                all_lines.extend(levelbuilder3d.build_obstacle(obs, self.current_level, self.player))

        all_lines.extend(levelbuilder3d.get_player_shape(self.player, self.current_level))

        if config.Display.depth_shade:
            # sorry tank, I just think it's a cool option <3
            depth_shading = (8 * self.foresight / 10, self.foresight)
        else:
            depth_shading = None

        all_2d_lines = self.camera.project_to_surface(screen, all_lines, depth_shading=depth_shading)
        neon_lines = neon.NeonLine.convert_line2ds_to_neon_lines(all_2d_lines)

        self.neon_renderer.draw_lines(screen, neon_lines, extra_darkness_factor=extra_darkness_factor)

        if show_score:
            screen.blit(self.score_font.render(str(self.player.get_score()), False, neon.LIME), (20, 20))


class PauseMenu(main.GameMode):

    def __init__(self, loop, gameplay_mode: GameplayMode, gesture_detector):
        super().__init__(loop)
        self.selected_option_idx = 0
        self.gameplay_mode = gameplay_mode
        self.options = [
            ("continue", lambda: self.continue_pressed()),
            ("exit", lambda: self.exit_pressed())
        ]

        self.title_font = fonts.get_font(config.FontSize.title)
        self.option_font = fonts.get_font(config.FontSize.option)

        self.pause_timer = 0  # how long we've been paused

        self.gesture_detector = gesture_detector

    def on_mode_start(self):
        SoundManager.play('blip2')
        SoundManager.set_song_volume_multiplier(0.5)

    def on_mode_end(self):
        SoundManager.set_song_volume_multiplier(1.0)

    def update(self, dt, events, gesture_actions=None):
        self.pause_timer += dt

        # 處理鍵盤事件
        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key in keybinds.MENU_UP:
                    SoundManager.play('blip')
                    self.selected_option_idx = (self.selected_option_idx - 1) % len(self.options)
                elif e.key in keybinds.MENU_DOWN:
                    SoundManager.play('blip')
                    self.selected_option_idx = (self.selected_option_idx + 1) % len(self.options)
                elif e.key in keybinds.MENU_ACCEPT:
                    self.options[self.selected_option_idx][1]()
                    return
                elif e.key in keybinds.MENU_CANCEL:
                    self.continue_pressed()
                    return

        # 處理手勢動作
        if gesture_actions:
            if 'jump' in gesture_actions:
                SoundManager.play('blip')
                self.selected_option_idx = (self.selected_option_idx - 1) % len(self.options)
            elif 'slide' in gesture_actions:
                SoundManager.play('blip')
                self.selected_option_idx = (self.selected_option_idx + 1) % len(self.options)
            elif 'enter' in gesture_actions:
                self.options[self.selected_option_idx][1]()
                return
            elif 'stop' in gesture_actions:
                self.continue_pressed()
                return


    def continue_pressed(self):
        SoundManager.play('accept')
        self.loop.set_mode(self.gameplay_mode)

    def exit_pressed(self):
        SoundManager.play('blip2')
        self.loop.set_mode(main.MainMenuMode(self.loop, self.gesture_detector))

    def draw_to_screen(self, screen):
        # make the level underneath fade darker slightly after you've paused
        max_darkness = 0.333
        max_darkness_time = 0.1  # second
        current_darkness = utility_functions.lerp(self.pause_timer / max_darkness_time, 1, max_darkness)

        # drawing level underneath this menu
        self.gameplay_mode.draw_to_screen(screen, extra_darkness_factor=current_darkness)

        screen_size = screen.get_size()
        title_surface = self.title_font.render('PAUSE', True, neon.WHITE)

        title_size = title_surface.get_size()
        title_y = screen_size[1] // 3 - title_size[1] // 2
        screen.blit(title_surface, dest=(screen_size[0] // 2 - title_size[0] // 2, title_y))

        option_y = max(screen_size[1] // 2, title_y + title_size[1])
        for i in range(len(self.options)):
            option_text = self.options[i][0]
            is_selected = i == self.selected_option_idx
            color = neon.WHITE if not is_selected else neon.RED

            option_surface = self.option_font.render(option_text.upper(), True, color)
            option_size = option_surface.get_size()
            screen.blit(option_surface, dest=(screen_size[0] // 2 - option_size[0] // 2, option_y))
            option_y += option_size[1]


class RetryMenu(main.GameMode):

    def __init__(self, loop, score, death_message, gesture_detector, gameplay_mode: GameplayMode):
        super().__init__(loop)
        self.score = score
        self.best_score = highscores.get_best()
        self.selected_option_idx = 0
        self.gameplay_mode = gameplay_mode
        self.options = [
            ("retry", lambda: self.retry_pressed(gesture_detector)),
            ("exit", lambda: self.exit_pressed())
        ]

        self.title_font = fonts.get_font(config.FontSize.title)
        self.option_font = fonts.get_font(config.FontSize.option)
        self.info_font = fonts.get_font(config.FontSize.info)

        self.death_message = death_message

        self.pause_timer = 0  # how long we've been paused
        self.gesture_detector = gesture_detector

    def on_mode_start(self):
        SoundManager.set_song_volume_multiplier(0.5)

    def on_mode_end(self):
        SoundManager.set_song_volume_multiplier(1.0)

    def update(self, dt, events, gesture_actions=None):
        self.pause_timer += dt

        # 處理鍵盤事件
        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key in keybinds.MENU_UP and self.pause_timer > 0.5:
                    SoundManager.play('blip')
                    self.selected_option_idx = (self.selected_option_idx - 1) % len(self.options)
                    self.pause_timer = 0  # 重置計時器
                elif e.key in keybinds.MENU_DOWN and self.pause_timer > 0.5:
                    SoundManager.play('blip')
                    self.selected_option_idx = (self.selected_option_idx + 1) % len(self.options)
                    self.pause_timer = 0  # 重置計時器
                elif e.key in keybinds.MENU_ACCEPT:
                    SoundManager.play('accept')
                    self.options[self.selected_option_idx][1]()
                    return
                elif e.key in keybinds.MENU_CANCEL:
                    SoundManager.play('blip2')
                    self.exit_pressed()
                    return

        # 處理手勢動作
        if gesture_actions:
            if 'jump' in gesture_actions and self.pause_timer > 0.5:
                SoundManager.play('blip')
                self.selected_option_idx = (self.selected_option_idx - 1) % len(self.options)
                self.pause_timer = 0  # 重置計時器
            elif 'slide' in gesture_actions and self.pause_timer > 0.5:
                SoundManager.play('blip')
                self.selected_option_idx = (self.selected_option_idx + 1) % len(self.options)
                self.pause_timer = 0  # 重置計時器
            elif 'enter' in gesture_actions:
                SoundManager.play('accept')
                self.options[self.selected_option_idx][1]()
                return
            elif 'stop' in gesture_actions:
                SoundManager.play('blip2')
                self.exit_pressed()
                return

    def retry_pressed(self, gesture_detector):
        self.loop.set_mode(GameplayMode(self.loop, gesture_detector))

    def exit_pressed(self):
        self.loop.set_mode(main.MainMenuMode(self.loop, self.gesture_detector))

    def draw_to_screen(self, screen):
        # make the level underneath fade darker slightly after you've paused
        max_darkness = 0.333
        max_darkness_time = 0.1  # second
        current_darkness = utility_functions.lerp(self.pause_timer / max_darkness_time, 1, max_darkness)

        # TODO fade underlying level to a color, for coolness

        # drawing level underneath this menu
        self.gameplay_mode.draw_to_screen(screen, extra_darkness_factor=current_darkness, show_score=False)

        screen_size = screen.get_size()

        title_surface = self.title_font.render('GAME OVER', True, neon.WHITE)
        title_size = title_surface.get_size()
        title_y = screen_size[1] // 3 - title_size[1] // 2
        screen.blit(title_surface, dest=(screen_size[0] // 2 - title_size[0] // 2, title_y))
        cur_y = title_y + int(title_size[1] * 0.9)

        death_msg_surface = self.info_font.render(self.death_message.upper(), True, neon.WHITE)
        death_msg_size = death_msg_surface.get_size()
        screen.blit(death_msg_surface, dest=(screen_size[0] // 2 - death_msg_size[0] // 2, cur_y))
        cur_y += int(death_msg_size[1] * 2)

        subtitle_surface1 = self.info_font.render("SCORE: {}".format(self.score), True, neon.WHITE)
        subtitle_surface1_size = subtitle_surface1.get_size()
        screen.blit(subtitle_surface1, dest=(screen_size[0] // 2 - subtitle_surface1_size[0] // 2, cur_y))
        cur_y += subtitle_surface1_size[1]

        subtitle_surface2 = self.info_font.render("BEST: {}".format(self.best_score), True, neon.WHITE)
        subtitle_surface2_size = subtitle_surface2.get_size()
        screen.blit(subtitle_surface2, dest=(screen_size[0] // 2 - subtitle_surface2_size[0] // 2, cur_y))
        cur_y += int(subtitle_surface2_size[1] * 2)

        option_y = max(screen_size[1] // 2, cur_y)
        for i in range(len(self.options)):
            option_text = self.options[i][0]
            is_selected = i == self.selected_option_idx
            color = neon.WHITE if not is_selected else neon.RED

            option_surface = self.option_font.render(option_text.upper(), True, color)
            option_size = option_surface.get_size()
            screen.blit(option_surface, dest=(screen_size[0] // 2 - option_size[0] // 2, option_y))
            option_y += option_size[1]
